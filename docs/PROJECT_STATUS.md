# Breast Cancer Detection System — Project Status & History

**Task:** malignancy (cancer) detection on Duke **BCS-DBT** 3D digital breast tomosynthesis.
**Approach:** per-slice 2D Swin-B → MIL aggregation, following *"Detection of breast cancer in
DBT with vision transformers"* (Sci Rep/Nature 2024), with imbalance handling and a
SIFT-DBT-style self-supervised pretraining stage.

**Current state (2026-07-01):** Phase A (supervised) complete at **AUROC 0.732**;
Phase B (self-supervised pretraining) **running** (~15 hr).

---

## 1. Environment & infrastructure ✅

Migrated from an old RTX 4090 box to this **A6000 server** (`oem@…`).

- **Host:** 1× RTX A6000 (48 GB, Ampere SM 8.6), driver 580.65.06 / CUDA 13.0, 64 cores, 251 GB RAM, Ubuntu 22.04.5.
- **Docker:** retargeted `Dockerfile` (`TORCH_CUDA_ARCH_LIST="8.6"`), `docker-compose.yml` (container `bc_detection_system_a6000`, `shm_size 64gb`). Installed Docker Engine + **NVIDIA Container Toolkit** (was missing → fixed the `nvidia` runtime).
- **Image built & verified:** torch 2.12.0+cu130, CUDA available, A6000 visible, monai/pydicom/timm/cv2 stack working.
- **Note:** `oem` joined the `docker` group post-session-start, so commands run via `sg docker -c '...'`.

## 2. Data pipeline ✅

Datasets on external SSD `/mnt/external_ssd/BreastCancer Datasets` (BCS-DBT, CBIS-DDSM, CMMD, Vindr-Mammo), mounted read-only into the container at `/data/datasets`.

- **BCS-DBT format:** each view = **one multiframe DICOM** = a `[Z,H,W]` volume (19,148 train volumes; download verified complete, ~1.4 TB).
- **Official metadata** (`BCS-DBT/Metadata/`): labels, boxes, file-paths CSVs (train/val/test).
- **`BCSDBTDataset`** (`data_preprocessing.py`): index built by merging file-paths+labels CSVs, parsing `classic_path` → glob the on-disk `.dcm`; cached as JSON.
- **Preprocessing** (`MedicalImagePreprocessor`): VOI-LUT + MONOCHROME1 fix + volume-wide normalize → MIP-based breast crop → per-slice CLAHE + aspect-preserving resize/pad to **384×384**.

### 2a. SSD preprocessing cache ✅
OS disk is 94% full, so caching goes to the **external SSD**:
- Writable mount `/mnt/external_ssd/bc_cache → /data/cache`.
- Each volume cached as compressed `uint8 [Z,384,384]` img+mask `.npz`.
- **Cache hit ≈ 0.04 s** vs ~14 s cold decode (317× faster).
- **`prewarm_cache.py`** decoded all **22,032 volumes** (train+val+test) → **77 GB cache**, integrity-verified.

## 3. Dataset label distribution (per view)

| Split | Normal | Actionable | Benign | **Malignant** | Total |
|-------|-------:|-----------:|-------:|--------------:|------:|
| Train | 18,232 | 716 | 124 | **76 (0.40%)** | 19,148 |
| Val   | 928 | 160 | 38 | **37 (3.18%)** | 1,163 |
| Test  | 1,356 | 244 | 61 | **60 (3.49%)** | 1,721 |

- **Positive = Malignant (Cancer) only** — a *cancer* detector (not lesion detection).
- **The core challenge: only 76 unique malignant train volumes** → extreme imbalance, and the structural performance ceiling. (See `dataset_label_distribution.md`.)

## 4. Model (`model.py` — `DimensionAgnostic3DSwin`)

- **Backbone:** Swin-B (`swin_base_patch4_window12_384`), grayscale `in_chans=1`, ImageNet-pretrained (or SSL via `ssl_weights=`).
- **Partial fine-tune:** patch_embed + stages 0–1 + heads trainable (~2.65 M); stages 2–3 frozen (~84.75 M).
- **Forward:** flatten `[B,Z]` → valid slices only → chunked backbone (slice_chunk=32, gradient checkpointing) → returns **`logits [B,Z]`** + **`saliency [B,Z,384,384]`** (WSOL CAM via `forward_features → 1×1 conv → upsample → sigmoid`).
- **A6000 perf:** bf16 autocast, TF32, cuDNN benchmark, padding-free slice processing.

## 5. Class-imbalance handling — evolution

| Stage | Method | Outcome |
|-------|--------|---------|
| v1 | plain `shuffle` (uniform) | ~98% all-negative batches |
| v2 | `WeightedRandomSampler` oversampling (≈1:1) | balanced but repeats 76 positives |
| v3 | `UndersampleNormalSampler` (Normal→12%) + `pos_weight` | 0.40%→2.45% positive |
| **v4 (final)** | **`BalancedBatchSampler`** — fixed ratio per batch + on-the-fly augmentation | guaranteed positives/batch |

Final training uses `BalancedBatchSampler` with **`pos_fraction=0.25` (1:3)** + **Focal Loss (α=0.75, γ=2)** + mild augmentation (hflip + brightness/contrast). `data_preprocessing.py` provides all three samplers.

## 6. Phase A — supervised MIL + WSOL ✅ (AUROC 0.732)

Implements the paper's plan + bounding-box localization (per Gemini "V2.1" directive).

- **Classification:** MIL **max-pool over slice logits** → **Focal Loss**.
- **Localization (WSOL):** **gated Dice loss** between per-slice saliency and GT box masks, applied **only on box-bearing slices** (box ≈ 2% of area → Dice, not BCE, to avoid saliency→0 collapse).
- **Inference:** scan probability = `max(moving_avg₈(sigmoid(slice probs)))` (paper's aggregation); box = threshold saliency on MIL-max slice → `cv2.boundingRect` (in 384-padded space).
- **Loss:** `L = FocalCls + 0.1 · Dice`. Early stopping (patience 4).

**Debugging note:** the first V2.1 config (1:7 + focal α=0.25 + λ=1.0) **collapsed to AUROC 0.53 / F1 0** (Dice dominated the loss; α=0.25 down-weighted the rare positive). Fixed with **1:3 + α=0.75 + λ=0.1** → recovered to **AUROC 0.732**.

**Paper alignment:** ✅ architecture (Swin-B@384) + slice→scan aggregation (moving-avg-8); **diverges** (intentionally) on cancer-only positive (paper = biopsied benign+cancer), balanced-batch sampling (paper = weighted loss), `in_chans=1`, frozen deep stages.

## 7. Phase B — SIFT-DBT self-supervised pretraining ⏳ (running)

**Goal:** break the 76-positive ceiling by learning DBT features from all ~19k *unlabeled* volumes, then fine-tune.

- **`pretrain_ssl.py`** — **MoCo v2** contrastive on cached slices (label-free).
  - **Positive pairs (DBT-specific):** cross-view (same study, different view, p=0.5) or neighbor slice (±9).
  - Swin-B encoder + 2-layer projection (128); momentum key encoder; queue K=4096, m=0.999, T=0.2.
  - Mild augmentation (no blur → preserves microcalcifications); AdamW + cosine, bf16, gradient checkpointing.
  - Saves **only the backbone** → `checkpoints/ssl_swinb_backbone.pth` (keys verified identical to the fine-tune model; strict-load OK).
- **Integration:** `DimensionAgnostic3DSwin(ssl_weights=...)` and `train_model(..., ssl_weights=...)` load it cleanly.
- **Status:** running detached (`ssl.log`), bs256, VRAM 41/49 GB, ~9 min/epoch → **~15 hr for 100 epochs**; backbone saved every 10 epochs.

## 8. Key files

| File | Role |
|------|------|
| `data_preprocessing.py` | `BCSDBTDataset`, preprocessing, SSD cache, collate, samplers |
| `model.py` | `DimensionAgnostic3DSwin` (Swin-B MIL + WSOL saliency) |
| `fine-tune.py` | Phase A training: focal + gated Dice, eval, early stopping, `ssl_weights` |
| `pretrain_ssl.py` | Phase B: MoCo v2 self-supervised pretraining |
| `prewarm_cache.py` | offline cache builder |
| `system_pipeline.md` | detailed learning/inference pipeline doc |
| `dataset_label_distribution.md` | label distribution tables |

## 9. Checkpoints

| File | What |
|------|------|
| `best_v20_classonly_auroc0749.pth` | V2.0 classification-only baseline (AUROC 0.749) |
| `best_phaseA_wsol_auroc0732.pth` | Phase A supervised MIL+WSOL (AUROC 0.732) |
| `best_3d_foundation.pth` | latest best (= Phase A 0.732) |
| `ssl_swinb_backbone.pth` | (pending) Phase B SSL backbone |

## 10. Next steps

1. Let Phase B SSL finish (~15 hr); confirm `loss`↓ / `top1`↑ converge.
2. **Ceiling-break test:** fine-tune with `ssl_weights='checkpoints/ssl_swinb_backbone.pth'` → compare val AUROC vs the **0.732** Phase-A baseline.
3. If SSL helps: consider patch-level MIL (native resolution) and threshold calibration (F1 is low; AUROC is the primary metric).
4. Deferred: EMBED 2D pretraining (tables present, images not downloaded), CBIS-DDSM/CMMD/Vindr.

## 11. Lessons / gotchas

- **Don't log heavy stdout to the exfat SSD** under load → caused `OSError(14)`/crash; log to nvme, disable tqdm on non-TTY.
- **bf16 (not fp16)** on Ampere — avoids `-1e9` masking overflow, no GradScaler.
- **Imbalance is hyperparameter-fragile**: balanced sampling overfits; too-aggressive undersampling + mis-set focal collapses. The real fix is *more positive signal* (SSL/data), not sampler tuning.
- **The 76-positive ceiling is structural** — Phase A's ~0.73–0.75 is about the limit of supervised learning here.
