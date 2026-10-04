# Artifacts Manifest — OLD vs NEW

**Updated:** 2026-08-10

Separates the **NEW** pipeline (combined BCS-DBT + DBT-2026, lesion-presence label) from the
**OLD/former** pipeline (single BCS-DBT, retired malignant-only label). **Former files are
untouched** — this is a labeling manifest, nothing was moved or deleted.

Rule going forward: NEW artifacts use the `combined_*` / `*_lesion*` / `fine_tuning_combined/`
namespace. Everything else is OLD and frozen for reference.

---

## NEW (current — combined lesion pipeline)

| Kind | Path | Notes |
|---|---|---|
| Code | `fine_tuning_combined/` | dataset_dbt2026.py, combined.py, fine-tune_combined.py |
| Label spec | `LABEL_DEFINITIONS.md` | authoritative (abnormal = benign+malignant) |
| Checkpoint | `checkpoints/combined_25d_swinb_r1_pos10.pth` | Swin-B, 9:1, lesion label (run in progress) |
| Train log | `train_combined_swinb_lesion.log` | current correct run |
| Extraction log | `unzip_dbt2026.log` | DBT-2026 unzip |
| Dataset | `/data/datasets/DBT-2026/` + BCS-DBT | combined |
| Cache | `/data/cache/dbt2026/` | DBT-2026 preproc + indexes |

## OLD (former — frozen, do not modify)

**Checkpoints (single BCS-DBT / malignant-only or malignant fine-tunes):**
- `checkpoints/best_v20_classonly_auroc0749.pth` — classification-only baseline 0.749
- `checkpoints/best_phaseA_wsol_auroc0732.pth` — Phase A WSOL 0.732
- `checkpoints/best_ssl_unfreeze_auroc0702.pth` — SSL unfreeze 0.702
- `checkpoints/best_3d_foundation.pth`, `ssl_swinb_backbone.pth` — SSL backbone
- `checkpoints/best_25d_r1.pth` (0.7392), `best_25d_r1_pos05.pth` (0.7248), `best_25d_swins_r1_pos05.pth` (0.7822) — 2.5D malignant runs

**Transfer checkpoints (malignant-target 2D transfer):**
- `transfer_learning/{vindr,cmmd,inbreast}/checkpoints/best_transfer_{r1,swins_r1,kd_swinb_r1,kd_swins_r1}.pth`

**Result docs:** `RESULTS_2p5d_transfer.md`.

**Logs** organized under `logs/`:
- `logs/finetune_bcsdbt/` — single-BCS-DBT runs (`train*.log`, `train_25d_*.log`, `train_ssl_ft.log`)
- `logs/ssl_prewarm/` — `ssl.log`, `prewarm.log`
- `logs/transfer/` — all `transfer_*.log`
- `logs/combined/` — combined-pipeline runs (`train_combined_swinb*.log`, `combined_chain.log`)
- `logs/data/` — `unzip_dbt2026.log`
- Active run log stays in repo root until it completes (currently `train_combined_swinb_lesion_strat.log`).

**Retired label:** malignant-only (positive = Cancer). See `LABEL_DEFINITIONS.md` §4.
