# Discord as a Defense

Code, raw results and figures for **"Discord as a Defense: A Red-Team Evaluation of a Blind Spot in Multimodal Fusion for Trustworthy Agentic Systems"**
(TRUSTMORE 2026, co-located with IEEE BigData 2026).

**Author:** Aliyu Agboola Alege
**License:** MIT (see `LICENSE`)
**AI disclosure:** the code was cleaned and documented with the support of Claude Sonnet 5.5 (Anthropic) as a pair-programming assistant; the research design, experiments and conclusions are the author's own.

## Layout
- `tables_and_figures/` — the LaTeX tables and PNG figures used in the paper, built by `code/make_all.py` from the CSVs in `results/`
- `code/` — `data.py` (digits-word, mnist-views; MNIST auto-download), `data_halves.py` (complementary-views benchmark), `models.py`, `models_extra.py` (PDF port, Kim-style gate, odd-one-out), `exp_v2.py`, `exp_v3.py`, `ablation_loss_terms.py`, `make_all.py`, `summarize.py`
- `results/` — raw long-format CSVs. `v2_*` hold the core models (PoE, SheafAlign, DGSA, single-modality baselines, attacks, adversarial training); `v3_*` hold the re-initialised / corruption-trained soft gates and the published-style baselines; `ablation_loss_terms.csv` holds the loss-term ablation

## Reproduce (CPU is enough; run from `code/`)
    pip install -r ../requirements.txt
    python exp_v2.py <digits_word|mnist_views|halves> paper <seed> [adv]   # core models; seeds 0-4
    python exp_v2.py mnist_views full <seed> [adv]                         # full scale; seeds 0-2 (~7 min vanilla, ~16-30 min adv per seed on 1 core)
    python exp_v3.py <digits_word|mnist_views|halves> <seed>               # soft-gate variants + PDF / Kim-style / OOO baselines
    python ablation_loss_terms.py                                          # loss-term ablation (writes ../results/ablation_loss_terms.csv)
    python make_all.py                                                     # regenerate every table and figure in ../tables_and_figures/

`make_all.py` builds all tables (including the loss-term ablation table) and figures (including the loss-term ablation figure, `fig4_loss_ablation.png`) from the CSVs in `results/`. `OUT_DIR` overrides where they are written. `figB_detection.png` is generated but not used in the paper.

MNIST is downloaded on first use (GitHub mirror); set `MNIST_PKL` to use a local copy. `RESULTS_DIR` overrides the output folder of the experiment scripts.

## Provenance and caveats
- Predictive Dynamic Fusion (`PDFLateFusion`) is a port of the late-fusion branch of https://github.com/Yinan-Xia/PDF. `KimGated` and `OOOFusion` are simplified re-implementations (no official code located); see the paper's Limitations.
- Hard-gate calibrations: `val30` (validation set with 30% corruption) and `clean95` (95th percentile of clean validation discord).
- The experiment scripts append rows to the CSVs in `results/`; set `RESULTS_DIR` to a scratch folder to re-run without touching the shipped results.

## Citation
    Aliyu Agboola Alege. "Discord as a Defense: A Red-Team Evaluation of a Blind Spot in Multimodal
    Fusion for Trustworthy Agentic Systems." TRUSTMORE 2026, IEEE BigData 2026.
