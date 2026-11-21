# Author: Aliyu Agboola Alege
# Developed with the support of Claude Sonnet 5.5 (Anthropic) as a pair-programming assistant.
# License: MIT (see ../LICENSE)
"""
Experiment harness for the core models (PoE, SheafAlign, DGSA hard/soft, single-modality baselines).

Features:
  * V-only and W-only single-modality baselines (matched architecture)
  * gate statistics: fire rate on CLEAN vs CORRUPTED examples
  * two hard-gate calibrations: 'val30' (sees 30% corruption on the validation set) and
    'clean95' (95th percentile of discord on CLEAN validation pairs; never sees corruption)
  * corruption of either modality: w (default), v, both-independent,
    both-coordinated (same wrong class in both -> low discord)
  * attacks that target each model's own FINAL output (incl. the gate), and a
    joint attack that also penalises discord (gate-aware)

Usage: python exp_v2.py <dataset: digits_word|mnist_views|halves> <scale: paper|full> <seed> [adv]
Results are appended (long format) to ../results/v2_<dataset>_<scale>[_adv].csv
"""
import sys, os, time, csv
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import DigitsWordDataset, MNISTViewsDataset, _corrupt_labels
from models import PoEModel, SheafBase, SheafModel, SheafHardGated, SheafSoftGated, EntropyGated

OUT = os.environ.get("RESULTS_DIR", os.path.join(HERE, "..", "results"))
os.makedirs(OUT, exist_ok=True)

CORR_RATES = [0.0, 0.15, 0.30, 0.45]
DIRECTIONS = ["w", "v", "both_indep", "both_coord"]
D_SHEAF, K_SHEAF, D_POE = 64, 32, 16
EPS, PGD_STEPS_EVAL, PGD_LR = 0.3, 20, 0.05
LAMBDAS = [0.0, 0.03, 0.3, 3.0]          # discord-penalty weights for the joint attack


# --------------------------------------------------------------------------- data
def build_dataset(name, scale, seed):
    if name == "halves":
        from data_halves import HalvesDataset
        sizes = dict(paper=(2000, 500, 750), full=(50000, 10000, 10000))[scale]
        ds = HalvesDataset(seed=seed, n_train=sizes[0], n_val=sizes[1], n_test=sizes[2])
        ds.cfg = dict(modb_key="edge", discrete=False, modb_dim=98)
        return ds
    if name == "digits_word":
        ds = DigitsWordDataset(seed=seed)
        cfg = dict(modb_key="y", discrete=True, modb_dim=0)
    else:
        sizes = dict(paper=(2000, 500, 750), full=(50000, 10000, 10000))[scale]
        ds = MNISTViewsDataset(seed=seed, n_train=sizes[0], n_val=sizes[1], n_test=sizes[2])
        cfg = dict(modb_key="edge", discrete=False, modb_dim=196)
    ds.cfg = cfg
    return ds


def common_kwargs(ds):
    return dict(vision_dim=ds.vision_dim, modb_dim=ds.cfg["modb_dim"], n_classes=ds.n_classes,
                modb_is_discrete=ds.cfg["discrete"], n_modb_classes=ds.n_classes)


def corrupt(ds, split, frac, seed, direction):
    """returns (x, w, mask). 'w' direction reproduces the original protocol exactly."""
    key = ds.cfg["modb_key"]
    w_c, mask = ds.make_corrupted(split, frac, seed)
    x = split["x"].clone()
    w = split[key].clone()
    y = split["y_np"]
    idx = np.where(mask.numpy())[0]
    if direction == "w":
        return x, w_c, mask
    rng = np.random.default_rng(seed + 31337)
    wrong_for_w = (y[idx] + rng.integers(1, 10, size=len(idx))) % 10
    wrong_for_v = (y[idx] + rng.integers(1, 10, size=len(idx))) % 10
    if direction == "both_coord":
        wrong_for_v = wrong_for_w
    # replace vision with a random TRAIN image of the wrong class
    if direction in ("v", "both_indep", "both_coord"):
        pool = {c: np.where(ds.train["y_np"] == c)[0] for c in range(ds.n_classes)}
        for i, c in zip(idx, wrong_for_v):
            x[i] = ds.train["x"][rng.choice(pool[c])]
    if direction in ("both_indep", "both_coord"):
        if ds.cfg["discrete"]:
            w[idx] = torch.tensor(wrong_for_w)
        else:
            for i, c in zip(idx, wrong_for_w):
                w[i] = torch.tensor(ds._edge_by_class[int(c)])
    return x, w, mask


# --------------------------------------------------------------------------- training
def train(model, data, epochs, modb_key, lr=2e-3, batch=None, loss_fn=None, seed=0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = data["x"].size(0)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        if batch is None or batch >= n:
            batches = [torch.arange(n)]
        else:
            batches = torch.randperm(n, generator=g).split(batch)
        for b in batches:
            opt.zero_grad()
            if loss_fn is None:
                loss = model.forward_train(data["x"][b], data[modb_key][b], data["y"][b])
            else:
                loss = loss_fn(model, data["x"][b], data[modb_key][b], data["y"][b])
            loss.backward()
            opt.step()
    return model


def vonly_loss(m, x, w, y):
    return F.cross_entropy(m.classifier(m.F_v(m.vision_enc(x))), y)


def wonly_loss(m, x, w, y):
    return F.cross_entropy(m.classifier(m.F_w(m.modb_enc(w))), y)



# --------------------------------------------------------------------------- adversarial training
ADV_FRAC, ADV_REGEN, PGD_STEPS_TRAIN = 0.30, 3, 5


def adv_modb(name, m, ds, x, y, rng):
    """white-box adversarial second-modality input against the model's own (final-output) loss."""
    f = final_logits_fn(name, m)
    if ds.cfg["discrete"]:
        best = y.clone(); best_loss = torch.full((len(y),), -1e9)
        with torch.no_grad():
            for c in range(ds.n_classes):
                cand = torch.full_like(y, c)
                loss = F.cross_entropy(f(x, cand), y, reduction="none")
                upd = (cand != y) & (loss > best_loss)
                best = torch.where(upd, cand, best); best_loss = torch.where(upd, loss, best_loss)
        return best
    wrong = torch.tensor([(int(c) + int(rng.integers(1, 10))) % 10 for c in y])
    w0 = torch.stack([torch.tensor(ds._edge_by_class[int(c)]) for c in wrong])
    w = w0.clone().requires_grad_(True)
    for _ in range(PGD_STEPS_TRAIN):
        g = torch.autograd.grad(F.cross_entropy(f(x, w), y, reduction="sum"), w)[0]
        with torch.no_grad():
            w += PGD_LR * g.sign()
            w.copy_(torch.min(torch.max(w, w0 - EPS), w0 + EPS).clamp(0, 1))
        w.requires_grad_(True)
    return w.detach()


def train_adv(name, model, data, epochs, key, ds, seed, batch=None, lr=2e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = data["x"].size(0)
    rng = np.random.default_rng(seed + 9000)
    g = torch.Generator().manual_seed(seed)
    cached = None
    for ep in range(epochs):
        batches = [torch.arange(n)] if (batch is None or batch >= n) else list(torch.randperm(n, generator=g).split(batch))
        for bi, b in enumerate(batches):
            x, w, y = data["x"][b], data[key][b].clone(), data["y"][b]
            regen = (batch is None and (ep % ADV_REGEN == 0)) or batch is not None
            if regen:
                k = int(ADV_FRAC * len(b))
                idx = torch.tensor(rng.choice(len(b), size=k, replace=False))
                cached = (idx, adv_modb(name, model, ds, x[idx], y[idx], rng))
            idx, adv_w = cached
            w[idx] = adv_w
            opt.zero_grad()
            model.forward_train(x, w, y).backward()
            opt.step()
    return model


# --------------------------------------------------------------------------- unified inference
@torch.no_grad()
def calibrate_val30(model, ds, val, seed):
    w_c, _ = ds.make_corrupted(val, 0.30, seed + 500)
    d = model.discord(val["x"], w_c)
    fv = model.F_v(model.vision_enc(val["x"]))
    fw = model.F_w(model.modb_enc(w_c))
    pv = model.classifier(fv).argmax(1)
    pb = model.classifier(model.fused_z(fv, fw)).argmax(1)
    best_thr, best_acc = None, -1
    for pct in range(10, 100, 5):
        thr = float(np.percentile(d.numpy(), pct))
        acc = (torch.where(d > thr, pv, pb) == val["y"]).float().mean().item()
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    return best_thr


@torch.no_grad()
def calibrate_clean(model, val, q=95):
    d = model.discord(val["x"], val[CUR["modb_key"]])
    return float(np.percentile(d.numpy(), q))


CUR = {}


def final_logits_fn(name, m, thr=None):
    """differentiable (w.r.t. w) logits of the model's own final output. For the hard gate
    the output is non-differentiable; we return the fused-path logits (BPDA-style surrogate)."""
    if name == "PoE":
        def f(x, w):
            mu, *_ = m.combine(x, w)
            return m.classifier(mu)
    elif name in ("SheafAlign", "DGSA-hard(val30)", "DGSA-hard(clean95)"):
        def f(x, w):
            h_v, h_w, fv, fw = m.encode(x, w)
            return m.classifier(m.fused_z(fv, fw))
    elif name == "DGSA-soft":
        def f(x, w):
            h_v, h_w, fv, fw = m.encode(x, w)
            d = ((fv - fw) ** 2).sum(-1)
            a = m.soft_alpha(d).unsqueeze(-1)
            return m.classifier(a * m.fused_z(fv, fw) + (1 - a) * fv)
    elif name == "Entropy":
        def f(x, w):
            h_v, h_w, fv, fw = m.encode(x, w)
            lv, lw = m.classifier_v(fv), m.classifier_w(fw)
            wt = F.softmax(torch.stack([-m._entropy(lv), -m._entropy(lw)], -1), -1)
            return wt[:, 0:1] * lv + wt[:, 1:2] * lw
    else:
        return None
    return f


def predict(name, M, x, w):
    """final prediction + gate-fired flag (nan-free bool tensor)"""
    m = M[name] if name in M else None
    nofire = torch.zeros(x.size(0), dtype=torch.bool)
    with torch.no_grad():
        if name == "PoE":
            return m.predict(x, w), nofire
        if name == "SheafAlign":
            return M["Sheaf"].predict(x, w), nofire
        if name.startswith("DGSA-hard"):
            s = M["Sheaf"]
            thr = M["thr_val30"] if "val30" in name else M["thr_clean95"]
            h_v, h_w, fv, fw = s.encode(x, w)
            d = ((fv - fw) ** 2).sum(-1)
            pv = s.classifier(fv).argmax(1)
            pb = s.classifier(s.fused_z(fv, fw)).argmax(1)
            g = d > thr
            return torch.where(g, pv, pb), g
        if name == "DGSA-soft":
            return m.predict(x, w), nofire
        if name == "Entropy":
            return m.predict(x, w), nofire
        if name == "V-only":
            s = M["V-only"]
            return s.classifier(s.F_v(s.vision_enc(x))).argmax(1), nofire
        if name == "W-only":
            s = M["W-only"]
            return s.classifier(s.F_w(s.modb_enc(w))).argmax(1), nofire
    raise KeyError(name)


MODEL_NAMES = ["PoE", "SheafAlign", "DGSA-hard(val30)", "DGSA-hard(clean95)", "DGSA-soft",
               "Entropy", "V-only", "W-only"]


def build_models(ds, seed, scale, adv=False):
    TR = (lambda nm, m, data, ep, key, **kw: train_adv(nm, m, data, ep, key, ds, kw.get('seed', 0), batch=kw.get('batch'))) if adv else None
    cfg = ds.cfg
    ck = common_kwargs(ds)
    train_d, val_d, _ = ds.get_splits()
    key = cfg["modb_key"]
    full = (scale == "full")
    bs = 2000 if full else None
    ep_poe, ep_sh = (60, 100) if full else (150, 300)
    M = {}
    if adv:
        torch.manual_seed(seed); M["PoE"] = train_adv("PoE", PoEModel(D=D_POE, **ck), train_d, ep_poe, key, ds, seed, batch=bs)
        torch.manual_seed(seed); M["Sheaf"] = train_adv("DGSA-hard(val30)", SheafHardGated(D=D_SHEAF, K=K_SHEAF, **ck), train_d, ep_sh, key, ds, seed, batch=bs)
        torch.manual_seed(seed); M["DGSA-soft"] = train_adv("DGSA-soft", SheafSoftGated(D=D_SHEAF, K=K_SHEAF, gate_temp=1.0, gate_midpoint=1.0, **ck), train_d, ep_sh, key, ds, seed, batch=bs)
    else:
        torch.manual_seed(seed); M["PoE"] = train(PoEModel(D=D_POE, **ck), train_d, ep_poe, key, batch=bs, seed=seed)
        torch.manual_seed(seed); M["Sheaf"] = train(SheafHardGated(D=D_SHEAF, K=K_SHEAF, **ck), train_d, ep_sh, key, batch=bs, seed=seed)
        torch.manual_seed(seed); M["DGSA-soft"] = train(SheafSoftGated(D=D_SHEAF, K=K_SHEAF, gate_temp=1.0, gate_midpoint=1.0, **ck), train_d, ep_sh, key, batch=bs, seed=seed)
    torch.manual_seed(seed); M["Entropy"] = train(EntropyGated(D=D_SHEAF, K=K_SHEAF, **ck), train_d, ep_sh, key, batch=bs, seed=seed)
    torch.manual_seed(seed); M["V-only"] = train(SheafBase(D=D_SHEAF, K=K_SHEAF, **ck), train_d, ep_sh, key, batch=bs, loss_fn=vonly_loss, seed=seed)
    torch.manual_seed(seed); M["W-only"] = train(SheafBase(D=D_SHEAF, K=K_SHEAF, **ck), train_d, ep_sh, key, batch=bs, loss_fn=wonly_loss, seed=seed)
    CUR["modb_key"] = key
    M["thr_val30"] = calibrate_val30(M["Sheaf"], ds, val_d, seed)
    M["thr_clean95"] = calibrate_clean(M["Sheaf"], val_d, 95)
    return M


# --------------------------------------------------------------------------- attacks
def attack_discrete(name, M, ds, x, y):
    """Exact worst case over the 9 wrong labels against the model's TRUE final output
    (including the gate): a sample counts as robust only if every wrong label leaves
    the prediction correct. Returns bool tensor 'robust'."""
    robust = torch.ones(x.size(0), dtype=torch.bool)
    for c in range(ds.n_classes):
        cand = torch.full_like(y, c)
        p, _ = predict(name, M, x, cand)
        ok = (p == y) | (cand == y)
        robust &= ok
    return robust


def model_of(name, M):
    return {"PoE": M["PoE"], "DGSA-soft": M["DGSA-soft"], "Entropy": M["Entropy"]}.get(name, M["Sheaf"])


def attack_pgd(name, M, ds, x, w0, y, lam, steps=PGD_STEPS_EVAL, eps=EPS, lr=PGD_LR):
    m = model_of(name, M)
    f = final_logits_fn(name, m)
    w = w0.clone().detach().requires_grad_(True)
    for _ in range(steps):
        loss = F.cross_entropy(f(x, w), y, reduction="sum")
        if lam > 0 and name not in ("PoE", "Entropy"):
            fv = m.F_v(m.vision_enc(x)).detach()
            fw = m.F_w(m.modb_enc(w))
            loss = loss - lam * ((fv - fw) ** 2).sum()
        g = torch.autograd.grad(loss, w)[0]
        with torch.no_grad():
            w += lr * g.sign()
            w.copy_(torch.min(torch.max(w, w0 - eps), w0 + eps).clamp(0, 1))
        w.requires_grad_(True)
    return w.detach()


def eval_attacks(M, ds, test, seed, names, frac=0.30):
    """Returns {(model, attack): corrupted-subset accuracy}. Attacks the corrupted-w subset."""
    rows = {}
    w_c, mask = ds.make_corrupted(test, frac, seed + 2000)
    idx = torch.where(mask)[0]
    x, y = test["x"][idx], test["y"][idx]
    if ds.cfg["discrete"]:
        for n in names:
            rows[(n, "worst-of-9")] = attack_discrete(n, M, ds, x, y).float().mean().item()
    else:
        w0 = w_c[idx]
        for n in names:
            if n in ("V-only",):
                p, _ = predict(n, M, x, w0)
                rows[(n, "any")] = (p == y).float().mean().item()
                continue
            if n == "W-only":
                continue
            ok_all = torch.ones(len(idx), dtype=torch.bool)
            per_lam = {}
            lams = [0.0] if n in ("PoE", "Entropy") else LAMBDAS
            for lam in lams:
                wa = attack_pgd(n, M, ds, x, w0, y, lam)
                p, _ = predict(n, M, x, wa)
                ok = (p == y)
                per_lam[lam] = ok.float().mean().item()
                ok_all &= ok
            rows[(n, "task-only(lam=0)")] = per_lam[0.0]
            if len(lams) > 1:
                rows[(n, "joint(worst-of-lams)")] = ok_all.float().mean().item()
    return rows



# --------------------------------------------------------------------------- detection / selective prediction
def entropy_of(logits):
    p = F.softmax(logits, -1)
    return -(p * torch.log(p + 1e-8)).sum(-1)


@torch.no_grad()
def score_and_pred(M, x, w):
    """per-detector (score, fused prediction). Higher score = more suspicious."""
    out = {}
    s, P = M["Sheaf"], M["PoE"]
    h_v, h_w, fv, fw = s.encode(x, w)
    zl = s.classifier(s.fused_z(fv, fw))
    out["discord"] = (((fv - fw) ** 2).sum(-1), zl.argmax(1))
    out["Sheaf-entropy"] = (entropy_of(zl), zl.argmax(1))
    mu, var, *_ = P.combine(x, w)
    pl = P.classifier(mu)
    out["PoE-variance"] = (var.mean(-1), pl.argmax(1))
    out["PoE-entropy"] = (entropy_of(pl), pl.argmax(1))
    return out


def detect_rows(M, ds, val, test, seed, base):
    from sklearn.metrics import roc_auc_score
    key = ds.cfg["modb_key"]
    ref = score_and_pred(M, val["x"], val[key])        # CLEAN validation pairs only
    thr = {k: float(np.percentile(v[0].numpy(), 95)) for k, v in ref.items()}
    rows = []
    for direction in DIRECTIONS:
        x, w, mask = corrupt(ds, test, 0.30, seed + 2000, direction)
        sc = score_and_pred(M, x, w)
        for k, (score, pred) in sc.items():
            acc_mask = score <= thr[k]
            ok = (pred == test["y"])
            rows.append(dict(base, kind="detect", direction=direction, rate=0.30, model=k,
                             auroc=roc_auc_score(mask.numpy(), score.numpy()),
                             coverage=acc_mask.float().mean().item(),
                             sel_acc=ok[acc_mask].float().mean().item() if acc_mask.any() else float("nan"),
                             recall=(~acc_mask & mask).float().sum().item() / max(mask.sum().item(), 1),
                             false_rej=(~acc_mask & ~mask).float().sum().item() / max((~mask).sum().item(), 1)))
    return rows



def riskcov_rows(M, ds, test, seed, base, grid=np.arange(0.50, 1.0001, 0.05)):
    rows = []
    for direction in DIRECTIONS:
        x, w, mask = corrupt(ds, test, 0.30, seed + 2000, direction)
        sc = score_and_pred(M, x, w)
        for k, (score, pred) in sc.items():
            ok = (pred == test["y"]).float().numpy()
            order = np.argsort(score.numpy(), kind="stable")
            for c in grid:
                n = max(int(round(c * len(order))), 1)
                rows.append(dict(base, kind="riskcov", direction=direction, rate=float(round(c, 2)), model=k,
                                 sel_acc=float(ok[order[:n]].mean())))
    return rows


# --------------------------------------------------------------------------- main
def main():
    dname, scale, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    ADV = len(sys.argv) > 4 and sys.argv[4] == "adv"
    t0 = time.time()
    ds = build_dataset(dname, scale, seed)
    train_d, val_d, test = ds.get_splits()
    M = build_models(ds, seed, scale, adv=ADV)
    print(f"[{dname}/{scale}/seed{seed}] trained in {time.time()-t0:.0f}s; thr_val30={M['thr_val30']:.1f} thr_clean95={M['thr_clean95']:.1f}", flush=True)
    rows = []
    base = dict(dataset=dname, scale=scale + ("_adv" if ADV else ""), seed=seed)

    # (1) corruption sweeps over directions
    for direction in DIRECTIONS:
        for frac in CORR_RATES:
            if frac == 0 and direction != "w":
                continue
            x, w, mask = corrupt(ds, test, frac, seed + 2000, direction)
            clean = ~mask
            for n in MODEL_NAMES:
                p, fired = predict(n, M, x, w)
                ok = (p == test["y"]).float()
                rows.append(dict(base, kind="sweep", direction=direction, rate=frac, model=n,
                                 overall=ok.mean().item(),
                                 clean=ok[clean].mean().item() if clean.any() else float("nan"),
                                 corrupted=ok[mask].mean().item() if mask.any() else float("nan"),
                                 fire_clean=fired[clean].float().mean().item() if clean.any() else float("nan"),
                                 fire_corrupt=fired[mask].float().mean().item() if mask.any() else float("nan")))
    # soft-gate alpha statistics (clean vs corrupted) in the 'w' direction at 30%
    x, w, mask = corrupt(ds, test, 0.30, seed + 2000, "w")
    with torch.no_grad():
        s = M["DGSA-soft"]
        d = s.discord(x, w)
        a = s.soft_alpha(d)
    rows.append(dict(base, kind="soft_alpha", direction="w", rate=0.30, model="DGSA-soft",
                     overall=a.mean().item(), clean=a[~mask].mean().item(), corrupted=a[mask].mean().item(),
                     fire_clean=s.midpoint.item(), fire_corrupt=s.log_temp.exp().item()))
    print(f"  sweeps done {time.time()-t0:.0f}s", flush=True)

    # (2) attacks on corrupted-w subset at 30%
    att = eval_attacks(M, ds, test, seed, ["PoE", "SheafAlign", "DGSA-hard(val30)", "DGSA-hard(clean95)",
                                           "DGSA-soft", "Entropy", "V-only", "W-only"])
    for (n, a_name), v in att.items():
        rows.append(dict(base, kind="attack", direction="w", rate=0.30, model=n, attack=a_name, corrupted=v))
    print(f"  attacks done {time.time()-t0:.0f}s", flush=True)

    rows += detect_rows(M, ds, val_d, test, seed, base)
    rows += riskcov_rows(M, ds, test, seed, base)
    path = os.path.join(OUT, f"v2_{dname}_{scale}{'_adv' if ADV else ''}.csv")
    keys = ["dataset", "scale", "seed", "kind", "direction", "rate", "model", "attack", "overall", "clean",
            "corrupted", "fire_clean", "fire_corrupt", "auroc", "coverage", "sel_acc", "recall", "false_rej"]
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        if new:
            wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in keys})
    print(f"  done {time.time()-t0:.0f}s, {len(rows)} rows -> {path}", flush=True)


if __name__ == "__main__":
    main()
