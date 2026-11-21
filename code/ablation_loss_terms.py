# Author: Aliyu Agboola Alege
# Developed with the support of Claude Sonnet 5.5 (Anthropic) as a pair-programming assistant.
# License: MIT (see ../LICENSE)
"""
Ablation: SheafAlign's loss = task + contrastive_alignment + reconstruction.
Which term causes the catastrophic (exactly 0.000) corrupted-subset collapse
on digits_word? Train 4 variants, holding architecture/capacity/epochs fixed
at the now-standardized D=64, K=32, 300 epochs.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
from data import DigitsWordDataset
from models import SheafBase

D, K, EPOCHS = 64, 32, 300
N_CLASSES = 10
SEEDS = [0, 1, 2]

variants = {
    "full (task+align+recon)":      dict(align=True,  recon=True),
    "task+align only (no recon)":   dict(align=True,  recon=False),
    "task+recon only (no align)":   dict(align=False, recon=True),
    "task only (no align, no recon)": dict(align=False, recon=False),
}

results = {name: [] for name in variants}

for seed in SEEDS:
    ds = DigitsWordDataset(seed=seed)
    train, val, test = ds.get_splits()
    common = dict(vision_dim=64, modb_dim=0, n_classes=N_CLASSES, modb_is_discrete=True, n_modb_classes=N_CLASSES)

    for name, cfg in variants.items():
        torch.manual_seed(seed)
        m = SheafBase(D=D, K=K, **common)
        opt = torch.optim.Adam(m.parameters(), lr=2e-3)
        for epoch in range(EPOCHS):
            opt.zero_grad()
            h_v, h_w, fv, fw = m.encode(train["x"], train["y"])
            z = m.fused_z(fv, fw)
            loss = F.cross_entropy(m.classifier(z), train["y"])
            if cfg["align"]:
                loss = loss + m.contrastive_alignment_loss(fv, fw)
            if cfg["recon"]:
                loss = loss + 0.5 * m.reconstruction_loss(h_v, h_w, fv, fw)
            loss.backward()
            opt.step()

        modb_corrupt, mask = ds.make_corrupted(test, 0.30, seed + 2000)
        with torch.no_grad():
            h_v, h_w, fv, fw = m.encode(test["x"], modb_corrupt)
            preds = m.classifier(m.fused_z(fv, fw)).argmax(dim=1)
            corrupted_acc = (preds[mask] == test["y"][mask]).float().mean().item()
            clean_acc = (preds[~mask] == test["y"][~mask]).float().mean().item()
        results[name].append((clean_acc, corrupted_acc))

# ---- save raw per-seed results so the paper's table and figure can be regenerated (see make_all.py)
import csv
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("RESULTS_DIR", os.path.join(HERE, "..", "results"))
os.makedirs(OUT, exist_ok=True)
csv_path = os.path.join(OUT, "ablation_loss_terms.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["variant", "seed", "clean_acc", "corrupted_acc"])
    for name, vals in results.items():
        for seed, (clean_acc, corrupted_acc) in zip(SEEDS, vals):
            w.writerow([name, seed, f"{clean_acc:.6f}", f"{corrupted_acc:.6f}"])
print(f"saved {csv_path}")

print(f"{'Variant':<36} | {'clean acc (mean)':>17} | {'corrupted acc (mean+-std)':>26}")
for name, vals in results.items():
    clean = np.mean([v[0] for v in vals])
    corr = np.array([v[1] for v in vals])
    print(f"{name:<36} | {clean*100:>16.1f}% | {corr.mean()*100:>20.1f}% +- {corr.std()*100:.1f}%")
