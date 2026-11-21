# Author: Aliyu Agboola Alege
# Developed with the support of Claude Sonnet 5.5 (Anthropic) as a pair-programming assistant.
# License: MIT (see ../LICENSE)
"""Fairer soft gates (re-initialised, corruption-trained) + published-style defence baselines.  python exp_v3.py <dataset> <seed>
Datasets: digits_word | mnist_views | halves  (paper scale).  Output: ../results/v3_<dataset>_paper.csv"""
import sys, os, time, csv
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, torch, torch.nn.functional as F
import exp_v2 as E
from models import SheafModel, SheafSoftGated
from models_extra import PDFLateFusion, KimGated, OOOFusion
from sklearn.metrics import roc_auc_score

OUT = os.environ.get("RESULTS_DIR", os.path.join(HERE, "..", "results"))
os.makedirs(OUT, exist_ok=True)
EPOCHS, LR, K_POOL, AUG = 300, 2e-3, 8, 0.30
NEW = ["DGSA-soft-init", "DGSA-soft-init+corr", "PDF", "Kim-gated(clean)", "Kim-gated(aug)", "OOO(Yang-style)"]


def pools(ds, train_d, seed):
    """fully-corrupted copies of the train set (all examples corrupted): w-corrupted and v-corrupted"""
    Wc = [E.corrupt(ds, train_d, 1.0, 5000 + 17 * seed + j, "w")[1] for j in range(K_POOL)]
    Xc = [E.corrupt(ds, train_d, 1.0, 9000 + 17 * seed + j, "v")[0] for j in range(K_POOL)]
    return Wc, Xc


def fit(model, step_loss, epochs=EPOCHS, seed=0):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(seed + 123)
    for ep in range(epochs):
        opt.zero_grad(); step_loss(ep, g).backward(); opt.step()
    return model


def mix(base, alt, frac, g):
    sel = torch.rand(base.size(0), generator=g) < frac
    out = base.clone(); out[sel] = alt[sel]
    return out, sel


def train_soft(ds, ck, tr, key, seed, corr, Wc):
    torch.manual_seed(seed)
    m = SheafSoftGated(D=E.D_SHEAF, K=E.K_SHEAF, gate_temp=1.0, gate_midpoint=1.0, **ck)
    x, w, y = tr["x"], tr[key], tr["y"]
    warm = EPOCHS // 3
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    g = torch.Generator().manual_seed(seed + 55)
    for ep in range(EPOCHS):
        if ep == warm:                              # data-driven gate init: m = clean-train discord 95th pct
            with torch.no_grad():
                d = m.discord(x, w)
                mid = float(np.percentile(d.numpy(), 95))
                m.midpoint.data.fill_(mid); m.log_temp.data.fill_(float(np.log(mid / 6)))
        opt.zero_grad()
        if ep < warm:
            loss = SheafModel.forward_train(m, x, w, y)
        else:
            w_in = mix(w, Wc[ep % K_POOL], AUG, g)[0] if corr else w
            h_v, h_w, fv, fw = m.encode(x, w_in)
            d = ((fv - fw) ** 2).sum(-1)
            a = m.soft_alpha(d).unsqueeze(-1)
            cls = F.cross_entropy(m.classifier(a * m.fused_z(fv, fw) + (1 - a) * fv), y)
            h_v0, h_w0, fv0, fw0 = m.encode(x, w)       # alignment / reconstruction on the CLEAN pairs
            loss = cls + m.contrastive_alignment_loss(fv0, fw0) + 0.5 * m.reconstruction_loss(h_v0, h_w0, fv0, fw0)
        loss.backward(); opt.step()
    return m


def build(ds, seed):
    ck = E.common_kwargs(ds); key = ds.cfg["modb_key"]
    tr, va, te = ds.get_splits()
    Wc, Xc = pools(ds, tr, seed)
    kw = dict(vision_dim=ds.vision_dim, modb_dim=ds.cfg["modb_dim"], n_classes=ds.n_classes,
              modb_is_discrete=ds.cfg["discrete"], n_modb_classes=ds.n_classes)
    x, w, y = tr["x"], tr[key], tr["y"]
    M = {}
    M["DGSA-soft-init"] = train_soft(ds, ck, tr, key, seed, False, Wc)
    M["DGSA-soft-init+corr"] = train_soft(ds, ck, tr, key, seed, True, Wc)
    torch.manual_seed(seed); m = PDFLateFusion(**kw); fit(m, lambda ep, g: m.loss(x, w, y), seed=seed); M["PDF"] = m
    torch.manual_seed(seed); m = KimGated(**kw); fit(m, lambda ep, g: m.loss(x, w, y), seed=seed); M["Kim-gated(clean)"] = m
    torch.manual_seed(seed); m = KimGated(**kw)
    fit(m, lambda ep, g: m.loss(x, mix(w, Wc[ep % K_POOL], AUG, g)[0], y), seed=seed); M["Kim-gated(aug)"] = m

    torch.manual_seed(seed); m = OOOFusion(**kw)

    def ooo_step(ep, g):
        u = torch.rand(x.size(0), generator=g)
        xi, wi = x.clone(), w.clone(); lab = torch.zeros(x.size(0), dtype=torch.long)
        sw = (u >= 0.70) & (u < 0.85); sv = u >= 0.85
        wi[sw] = Wc[ep % K_POOL][sw]; lab[sw] = 2
        xi[sv] = Xc[ep % K_POOL][sv]; lab[sv] = 1
        return m.loss(xi, wi, y, lab)
    fit(m, ooo_step, seed=seed); M["OOO(Yang-style)"] = m
    # thresholds for soft-gate fire statistics are not needed (continuous)
    return M


def predict_new(name, M, x, w):
    m = M[name]
    with torch.no_grad():
        if name.startswith("DGSA-soft"):
            return m.predict(x, w), None
        if name.startswith("OOO"):
            p, r = m.predict(x, w, return_route=True)
            return p, (r != 0)
        return m.predict(x, w), None


def logits_new(name, M):
    m = M[name]
    if name.startswith("DGSA-soft"):
        def f(x, w):
            h_v, h_w, fv, fw = m.encode(x, w)
            a = m.soft_alpha(((fv - fw) ** 2).sum(-1)).unsqueeze(-1)
            return m.classifier(a * m.fused_z(fv, fw) + (1 - a) * fv)
        return f
    return m.logits


def attack_pgd_new(name, M, x, w0, y, lam):
    m = M[name]; f = logits_new(name, M)
    w = w0.clone().detach().requires_grad_(True)
    for _ in range(E.PGD_STEPS_EVAL):
        loss = F.cross_entropy(f(x, w), y, reduction="sum")
        if lam > 0 and name.startswith("DGSA-soft"):
            fv = m.F_v(m.vision_enc(x)).detach(); fw = m.F_w(m.modb_enc(w))
            loss = loss - lam * ((fv - fw) ** 2).sum()
        g = torch.autograd.grad(loss, w)[0]
        with torch.no_grad():
            w += E.PGD_LR * g.sign()
            w.copy_(torch.min(torch.max(w, w0 - E.EPS), w0 + E.EPS).clamp(0, 1))
        w.requires_grad_(True)
    return w.detach()


def main():
    dname, seed = sys.argv[1], int(sys.argv[2]); t0 = time.time()
    ds = E.build_dataset(dname, "paper", seed)
    tr, va, te = ds.get_splits(); key = ds.cfg["modb_key"]
    M = build(ds, seed)
    print(f"[v3 {dname} seed{seed}] trained {time.time()-t0:.0f}s", flush=True)
    base = dict(dataset=dname, scale="paper", seed=seed); rows = []
    for direction in E.DIRECTIONS:
        for frac in [0.0, 0.30]:
            if frac == 0 and direction != "w": continue
            x, w, mask = E.corrupt(ds, te, frac, seed + 2000, direction)
            clean = ~mask
            for n in NEW:
                p, fired = predict_new(n, M, x, w); ok = (p == te["y"]).float()
                rows.append(dict(base, kind="sweep", direction=direction, rate=frac, model=n, overall=ok.mean().item(),
                                 clean=ok[clean].mean().item() if clean.any() else float("nan"),
                                 corrupted=ok[mask].mean().item() if mask.any() else float("nan"),
                                 fire_clean=fired[clean].float().mean().item() if fired is not None and clean.any() else float("nan"),
                                 fire_corrupt=fired[mask].float().mean().item() if fired is not None and mask.any() else float("nan")))
    # soft alpha stats
    x, w, mask = E.corrupt(ds, te, 0.30, seed + 2000, "w")
    for n in ["DGSA-soft-init", "DGSA-soft-init+corr"]:
        with torch.no_grad():
            s = M[n]; a = s.soft_alpha(s.discord(x, w))
        rows.append(dict(base, kind="soft_alpha", direction="w", rate=0.30, model=n, overall=a.mean().item(),
                         clean=a[~mask].mean().item(), corrupted=a[mask].mean().item()))
    # OOO detection AUROC
    for direction in E.DIRECTIONS:
        x, w, mask = E.corrupt(ds, te, 0.30, seed + 2000, direction)
        with torch.no_grad():
            sc = M["OOO(Yang-style)"].score(x, w)
        rows.append(dict(base, kind="detect", direction=direction, rate=0.30, model="OOO-score", auroc=roc_auc_score(mask.numpy(), sc.numpy())))
    # attacks on corrupted-w subset @30%
    w_c, mask = ds.make_corrupted(te, 0.30, seed + 2000)
    idx = torch.where(mask)[0]; x, y = te["x"][idx], te["y"][idx]
    for n in NEW:
        if ds.cfg["discrete"]:
            robust = torch.ones(len(idx), dtype=torch.bool)
            for c in range(ds.n_classes):
                cand = torch.full_like(y, c)
                p, _ = predict_new(n, M, x, cand)
                robust &= (p == y) | (cand == y)
            rows.append(dict(base, kind="attack", direction="w", rate=0.30, model=n, attack="worst-of-9", corrupted=robust.float().mean().item()))
        else:
            w0 = w_c[idx]; ok_all = torch.ones(len(idx), dtype=torch.bool); per = {}
            lams = [0.0, 0.03, 0.3, 3.0] if n.startswith("DGSA-soft") else [0.0]
            for lam in lams:
                wa = attack_pgd_new(n, M, x, w0, y, lam)
                ok = predict_new(n, M, x, wa)[0] == y; per[lam] = ok.float().mean().item(); ok_all &= ok
            rows.append(dict(base, kind="attack", direction="w", rate=0.30, model=n, attack="task-only(lam=0)", corrupted=per[0.0]))
            if len(lams) > 1:
                rows.append(dict(base, kind="attack", direction="w", rate=0.30, model=n, attack="joint(worst-of-lams)", corrupted=ok_all.float().mean().item()))
    keys = ["dataset", "scale", "seed", "kind", "direction", "rate", "model", "attack", "overall", "clean", "corrupted",
            "fire_clean", "fire_corrupt", "auroc"]
    path = os.path.join(OUT, f"v3_{dname}_paper.csv"); new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        if new: wr.writeheader()
        for r in rows: wr.writerow({k: r.get(k, "") for k in keys})
    print(f"  done {time.time()-t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
