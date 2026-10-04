# Results — index

All transfer-learning experiments complete (2026-09-01). This directory holds the data; analysis is left to the reader.

## Files
- **`auroc_per_seed.csv`** — raw per-run TEST AUROC (150 rows). Columns: `dataset, arm, method, seed, test_auroc, val_auroc, test_n, test_pos`.
- **`auroc_summary.csv`** — mean ± SD per (dataset, arm, method), n=5 (30 configs).
- **`RESULTS_transfer_learning.md`** — data dictionary + design note.
- Raw model outputs (labels + predicted probabilities per run): `../experiments/preds/*.json` — for own AUROC/DeLong/bootstrap.

## Conditions (methods)
- `m0_2donly` — ImageNet-init → transfer-learned to 2D (ImageNet-only).
- `m1_wt_llrd` — ImageNet → DBT fine-tune → transfer-learned to 2D (DBT additional).
- `mscratch` — random-init → DBT fine-tune → transfer-learned to 2D (DBT-only).

## Arms
- `natural` — dataset's own class ratio. `balanced` — 1:1 training sampler. Same models/test sets; only training sampler differs. RSNA is balanced-only (~2% prevalence).

## Datasets (all patient-grouped, leakage-safe, biopsy/pathology labels)
| dataset | region | task | arms present |
|---|---|---|---|
| embed_nat / embed | USA | lesion (normal vs benign+malignant) | natural (nat) + balanced (embed 50:50) |
| mias | UK | lesion | natural + balanced |
| cddcesm | Egypt | lesion | natural + balanced |
| cmmd | China | malignancy (benign vs malignant) | natural + balanced |
| rsna | USA multi-site | malignancy (cancer vs non-cancer) | balanced only |
| cbis | USA | malignancy (benign vs malignant) | natural + balanced |

Excluded (fail inclusion criterion, not in any result): INbreast (no patient IDs), BUSI (ultrasound), VinDr (no biopsy).

Coverage note: mscratch was run on natural arms (+ malignancy balanced); balanced lesion arms have m0/m1 only.

## Stage-1 DBT checkpoints (transfer sources)
- `../checkpoints_b16/combined_25d_swinb_r1_pos10_strat_pureN_b16.pth` — ImageNet→DBT. Held-out DBT TEST AUROC 0.929 (VAL 0.951).
- `../checkpoints_b16/combined_25d_swinbscratch_r1_pos10_strat_pureN_b16.pth` — scratch→DBT. Held-out DBT TEST AUROC 0.851 (VAL 0.894).

## Reproducibility
- Seeds {42,1,2,3,4} fixed (global RNG + DataLoader generator + per-worker). Transfer batch 16, LLRD, val-selection, TEST-once.
- Scripts: `../experiments/multiseed_matrix.py` (lesion natural), `../experiments/balanced_run.py` (lesion balanced), `../experiments/scratch_transfer.py` (mscratch lesion), `../experiments/malignancy_run.py` + `malignancy_balanced_run.py` (malignancy), `../fine_tuning_combined/train_scratch.py` (scratch DBT).
- Logs: `../logs/transfer/` (transfer), `../logs/combined/` (DBT fine-tune).
