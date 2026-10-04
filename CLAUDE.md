# CLAUDE.md — Breast Cancer Detection System (BCS-DBT)

## Main purpose
**The primary goal of this project is the reproducibility of medical-AI systems.** The breast-cancer
DBT→2D pipeline is the *vehicle* for that goal, not the end in itself. Every methodological choice is
made to be reproducible and auditable: fixed deterministic seeds (global RNG + DataLoader generator +
per-worker seeding), **no synthetic per-epoch random resampling** (fixed train sets), patient-grouped
leakage-safe splits (no CC/MLO / L-R / region contamination), explicit per-stage LLRD, and an authoritative,
disclosed label definition. Headline accuracy (e.g. TEST AUROC 0.929) is reported, but claims are
framed as *reproducible* results with stated limitations (caveated datasets, single-seed estimates), not
leaderboard numbers.

## What this is (the vehicle)
**Lesion-presence classification** on 3D digital breast tomosynthesis, trained on a
**combined BCS-DBT + DBT-2026** dataset. Per-slice 2D **Swin** → MIL max-pool → scan/case
aggregation, then **DBT→2D transfer learning** (VinDr/CMMD/INbreast). Runs in Docker on an
**RTX A6000** host. Task = **lesion (abnormal) detection** — abnormal = benign-lesion + malignant;
*not* box-level object detection. See `docs/LABEL_DEFINITIONS.md`.

## Repository layout (organized 2026-08-25)
Root holds only `CLAUDE.md`, build config (`Dockerfile`, `docker-compose.yml`, `requirements.txt`),
the code packages, and any **active** run log. Everything else is foldered:
- `common/` `fine_tuning/` `fine_tuning_combined/` `transfer_learning/` — code (see Files table).
- `docs/` — all documentation (`PROJECT_DOSSIER.md`, `PAPER_TABLES.md`, `LABEL_DEFINITIONS.md`,
  `PROJECT_STATUS.md`, `HANDOFF_papers.md`, `system_pipeline.md`, etc.) + `Claude Corporated Doc.xlsx`.
- `logs/` — run logs by kind: `logs/combined/` (DBT fine-tune), `logs/transfer/` (2D transfer),
  `logs/data/` (EMBED pull + manifests), `logs/markers/` (`.done` sentinels).
- `papers/` — reference PDFs + figures (`image.png`).
- `scripts/` — one-off helpers (`embed_finish.sh`, `smoke_test_25d.py`).
- `checkpoints/`, `checkpoints_b16/` — model checkpoints (never move/rename best ones).

## Environment & how to run
- **Host:** user `oem`, 1× RTX A6000 (48 GB, CC 8.6), CUDA 13.0, 64 cores, 251 GB RAM, Ubuntu 22.04.
- **Docker is required; `oem` isn't in the docker group in pre-existing shells → prefix every docker call with `sg docker -c '...'`.** Container: `bc_detection_system_a6000`.
- Lifecycle (from project root): `sg docker -c 'docker compose up -d | down | logs -f'`.
- Run code in-container: `sg docker -c 'docker exec -w /workspace bc_detection_system_a6000 python <script>.py'`.
- Long jobs: launch detached (`docker exec -d ... sh -c "python -u X.py > /workspace/X.log 2>&1"`) and poll the log.

## Data & cache (critical paths)
- Datasets mounted read-only at `/data/datasets` (host `/mnt/external_ssd/BreastCancer Datasets`). BCS-DBT root = `/data/datasets/BCS-DBT`.
- Each view = **one multiframe DICOM = one `[Z,H,W]` volume** (NOT per-slice files).
- **Preprocessed cache on the SSD at `/data/cache`** (host `/mnt/external_ssd/bc_cache`) — `preproc/<split>/<PID>/<StudyUID>_<view>.npz` (uint8 `[Z,384,384]` img+mask) + `bcsdbt_<split>_index_v2.json`. All 22,032 volumes pre-warmed (~77 GB). Cache hit ≈ 0.04 s.
- **OS disk `/` is ~94% full — never cache there. Everything heavy goes on the external SSD.**

## Files
| File | Role |
|------|------|
| `data_preprocessing.py` | `BCSDBTDataset`, preprocessing, SSD cache, `collate_dbt_volumes`, samplers (`make_balanced_batch_sampler`, `make_undersample_sampler`), augmentation |
| `model.py` | `DimensionAgnostic3DSwin` — per-slice Swin-B (in_chans=1) + MIL + WSOL saliency; `ssl_weights=` + `freeze_backbone=` |
| `fine-tune.py` | training: focal + gated-Dice, LLRD, bf16, aggregation, early stop, `ssl_weights` |
| `pretrain_ssl.py` | Phase B: MoCo v2 SSL (cross-view/±9 positives) on Swin → `ssl_swinb_backbone.pth` |
| `prewarm_cache.py` | offline cache builder |
| docs (in `docs/`) | `PROJECT_DOSSIER.md`, `PAPER_TABLES.md`, `PROJECT_STATUS.md`, `프로젝트_전체_정리.md`, `system_pipeline.md`, `dataset_label_distribution.md`, `HANDOFF_papers.md` |

## Hard-won conventions & gotchas
- **bf16 autocast, not fp16** (Ampere) — avoids the `-1e9` masking overflow; no GradScaler.
- **Do NOT redirect heavy stdout to a log file on the exfat SSD** — tqdm spam there caused `OSError(14)` and killed a job. Log to nvme (`/workspace`); use `tqdm(..., disable=None)` for non-TTY.
- Padded slices are masked with `-inf` (bf16-safe), not `-1e9`.
- Cache paths must match exactly: `cache_dir='/data/cache'` (a wrong path silently re-decodes ~8 h).
- A6000 perf: TF32, `cudnn.benchmark`, `matmul_precision('high')`, `pin_memory`, `non_blocking`, `num_workers≈12`.
- Confirm/ask before killing a running job or overwriting a checkpoint; preserve best checkpoints under descriptive names.

## Label definition — see `docs/LABEL_DEFINITIONS.md` (authoritative)
- **Task = lesion presence.** **abnormal (yes-lesion) = benign-lesion + malignant**; **normal = no-lesion**. Report ratios as **normal:abnormal**.
- **BCS-DBT:** abnormal = `benign + cancer`; normal = `Normal`; **Actionable is dropped (excluded)**.
- **DBT-2026 (Segmed):** abnormal = Group `A/C/D`; normal = Group `B`.
- **Combined (BCS-DBT + DBT-2026):** train normal:abnormal = **18,488:1,502**; sampler undersamples normal to **9:1**. This ~20× more abnormal signal (1,502 vs old 76) is the point of the merge.

## Current state (2026-07)
- **Phase A** (supervised MIL + WSOL, Swin, ImageNet init): best val **AUROC 0.732**; earlier classification-only baseline **0.749**.
- **Phase B** SSL (MoCo on Swin): done, contrastive top1 0.88 → `ssl_swinb_backbone.pth`.
- **Ceiling-break** (SSL init + full unfreeze + LLRD): **underperformed (0.7015, overfit)** — did not beat 0.732.
- Checkpoints: `best_v20_classonly_auroc0749.pth`, `best_phaseA_wsol_auroc0732.pth`, `ssl_swinb_backbone.pth`, `best_ssl_unfreeze_auroc0702.pth`.

## Papers in scope (PDFs in `papers/`; see `docs/HANDOFF_papers.md` for the full analysis)
- `papers/s41598-024-72707-2.pdf` — Swin classification + moving-avg(8)→max→CC/MLO aggregation.
- `papers/2403.13148v1.pdf` (SIFT-DBT) — MoCo v2 SSL + patch-MIL; repo `github.com/XYPB/SIFT_DBT`.
- `papers/3D_Self-Supervised…Brain_Tumor…pdf` — 3D MAE → 3D→2D **deflation** → 2D ViT (brain MRI).
