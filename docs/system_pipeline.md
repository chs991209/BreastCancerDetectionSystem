# BCS-DBT Cancer Detection — Learning Pipeline & Prediction Format

System for **malignancy (cancer) detection** on Duke BCS-DBT 3D digital breast
tomosynthesis volumes. Architecture/inference follow *"Detection of breast cancer
in digital breast tomosynthesis with vision transformers"* (Sci Rep / Nature 2024):
a 2D Swin Transformer scores each slice, and slice scores are aggregated into one
scan-level probability.

Files: `data_preprocessing.py` (data), `model.py` (network), `fine-tune.py` (training),
`prewarm_cache.py` (offline cache build).

---

## 0. End-to-end flow

```
DICOM volume (1 file = 1 view = [Z,H,W])
   └─ read_volume         VOI-LUT · MONOCHROME1 fix · volume-wide 0–255 normalize
   └─ compute_crop_rect   MIP over Z → Otsu → largest contour (breast bbox)
   └─ per slice z:        crop → CLAHE → aspect-preserve resize+pad → 384×384
   └─ cache .npz (uint8 [Z,384,384] img+mask) on SSD            ← built once
        │  (next epochs load in ~0.04 s instead of ~14 s decode)
        ▼
   __getitem__  → float [1,Z,384,384] /255  → (train) augment
        ▼
   collate      → pad variable Z to Z_max  → volumes [B,1,Z_max,H,W], depth_mask [B,Z_max]
        ▼
   model        → per-slice 2D Swin-B + head → slice logits [B,Z] (pad = −inf)
        ▼
   TRAIN: scan_logit = max_z(logits)  → BCEWithLogitsLoss
   INFER: P_scan = max_z( moving_avg_8( sigmoid(logits) ) )  ∈ [0,1]
```

---

## 1. Data preprocessing / input manipulation

Per view, BCS-DBT stores the **whole volume as one multiframe DICOM**
(`pixel_array` = `[Z, H, W]`). `MedicalImagePreprocessor` (in `data_preprocessing.py`):

1. **`read_volume(path)`** → normalized `uint8 [Z, H, W]`
   - `pydicom.dcmread` (JPEG2000/JPEG-LS decoded via gdcm/pylibjpeg)
   - `apply_voi_lut` (windowing); falls back to raw if tags missing
   - `MONOCHROME1` → invert to MONOCHROME2 polarity
   - normalize **over the whole volume** to 0–255 (slice-to-slice brightness consistency)

2. **`compute_crop_rect(volume)`** — one crop box for the whole volume
   - Max-Intensity-Projection over Z → Gaussian blur → Otsu threshold →
     largest external contour → bounding rect (the breast region). Applied
     identically to every slice so the Z-stack stays spatially aligned.

3. **Per-slice finalize** (`finalize_image` / `finalize_mask`)
   - crop to breast rect
   - **CLAHE** (`clipLimit=2.0`, `tileGridSize=8×8`) — local contrast, image only
   - aspect-ratio-preserving resize + zero-pad to **384×384**
     (`INTER_AREA` for image, `INTER_NEAREST` for mask)
   - lesion mask rendered from the boxes CSV (`X,Y,W,H` per slice); **currently
     prepared but NOT used by the loss** — training is classification-only.

4. **Tensors** (`__getitem__`): `uint8 [Z,384,384]` → `float [1,Z,384,384] / 255`
   for both image and mask. Returns `(volume_tensor, mask_volume, cancer_label)`.

### 1a. Preprocessed cache (SSD)
First access decodes + preprocesses, then saves `uint8 [Z,384,384]` img+mask as a
compressed `.npz` to `/data/cache/preproc/<split>/<PID>/<StudyUID>_<view>.npz`
(atomic write). All 22,032 volumes pre-warmed (~77 GB on external SSD; OS disk
untouched). Cache hit ≈ **0.04 s** vs ~14 s cold decode.

### 1b. Train-time augmentation (`augment=True`, train only)
Applied on the float tensor after load, per sample (so the few unique positives
are never identical across repeats):
- random **horizontal flip** (image + mask together)
- **brightness/contrast jitter** (image only; contrast ×[0.9,1.1], brightness ±0.05)

---

## 2. Dataset, labels, and class balance

- **Index**: built by merging BCS-DBT `file-paths` + `labels` CSVs, parsing
  `classic_path` → `(PID, StudyUID, SeriesUID)` → glob the on-disk `.dcm`.
  Cached as JSON. Each entry stores the 4-class `klass`
  (`normal/actionable/benign/cancer`) and the binary target.
- **Positive label = `Cancer` (malignant) only.** Normal + Actionable + Benign
  are negatives (Benign/Actionable act as *hard negatives*).
- **Distribution (per view):**

  | Split | Normal | Actionable | Benign | Malignant | Total |
  |-------|-------:|-----------:|-------:|----------:|------:|
  | Train | 18,232 | 716 | 124 | **76 (0.40%)** | 19,148 |
  | Val   | 928 | 160 | 38 | **37 (3.18%)** | 1,163 |
  | Test  | 1,356 | 244 | 61 | **60 (3.49%)** | 1,721 |

- **Imbalance handling — `BalancedBatchSampler`** (guaranteed per-batch balance):
  - every batch = fixed **`pos_fraction` split** (default 0.5 → `batch_size=4` ⇒ 2 malignant + 2 negative)
  - negatives per epoch = **100% of Actionable+Benign (840)** + **12% of Normal (2,188)**, reshuffled each epoch
  - 76 malignant volumes **cycled** (reshuffled when exhausted), each pass differently augmented
  - epoch = **1,514 batches** (≈6,056 volumes)
  - because batches are balanced, the loss uses **plain BCE** (no `pos_weight`)

### 2a. Collate (`collate_dbt_volumes`)
Volumes have different Z, so the batch is zero-padded to `Z_max`:
- `volumes [B,1,Z_max,H,W]`, `masks [B,1,Z_max,H,W]`
- `depth_mask [B,Z_max]` (True = real slice, False = padding)
- `labels [B]`

---

## 3. Model (`DimensionAgnostic3DSwin`)

- **Backbone**: `timm` `swin_base_patch4_window12_384` (Swin-B), `in_chans=1`
  (grayscale), `num_classes=0` (feature extractor), ImageNet-pretrained.
- **Head**: `Linear(feat→feat/2) → GELU → Dropout(0.3) → Linear(feat/2→1)`.
- **Partial fine-tuning**: trainable = patch_embed + Swin stages 0–1 + head
  (**~2.65 M**); stages 2–3 frozen (**~84.75 M**). (VRAM/overfit control; the
  paper instead fine-tunes all layers.)
- **`forward(volumes [B,1,Z,H,W], depth_mask [B,Z]) → slice logits [B,Z]`**
  1. flatten to `[B*Z, 1, H, W]`, **keep only valid (non-padded) slices**
  2. run them through the backbone in **chunks of `slice_chunk=32`**
     (gradient checkpointing during training → bounded VRAM)
  3. head → per-slice logit; **scatter back to `[B,Z]`, padding = `−inf`**
     (so padding is ignored by both `max` and `sigmoid`)

---

## 4. Training configuration (`fine-tune.py`)

| Item | Value |
|------|-------|
| Loss | `BCEWithLogitsLoss` (plain — batches already balanced) |
| Aggregation for loss | **MIL max-pool**: `scan_logit = max_z(slice_logits)` |
| Optimizer | `AdamW`, lr `1e-4`, weight_decay `1e-2` (trainable params only) |
| LR schedule | `CosineAnnealingLR(T_max=epochs)` |
| Mixed precision | **bf16 autocast** (Ampere-native; no GradScaler) |
| Grad clip | `clip_grad_norm_(max_norm=1.0)` |
| A6000 perf | TF32 on, `cudnn.benchmark`, `matmul_precision('high')`, `pin_memory`, `non_blocking` H2D, `num_workers=12` |
| Epochs / batch | 15 / 4 |
| Checkpoint | best **val AUROC** → `checkpoints/best_3d_foundation.pth` |

Train step: `autocast(bf16)` → forward → `max_z` → BCE → `backward` → clip → `step`.

---

## 5. Detection / prediction result format

The system outputs a **scan-level (per-view) malignancy probability**, not boxes.

**Raw model output:** per-slice logits `[B, Z]` (padding = `−inf`).

**Inference aggregation** (paper's method, `fine-tune.py` validation):
1. `p_z = sigmoid(slice_logits)` → per-slice malignancy probability `[B, Z]`
2. take valid slices `p_z[:valid_z]`
3. **moving average, window = 8** (`avg_pool1d(kernel_size=8, stride=1)`)
   — smooths over neighboring slices; if `valid_z < 8`, fall back to plain max
4. **`P_scan = max(smoothed)`** ∈ **[0, 1]** — the scan's malignancy score
5. **Binary decision**: `malignant = P_scan > 0.5`

**So per input view you get:**

| Output | Shape / type | Meaning |
|--------|--------------|---------|
| `P_scan` | scalar float ∈ [0,1] | malignancy probability of the view |
| `pred` | {0,1} | `P_scan > 0.5` |
| `p_z` (per-slice) | `[Z]` floats | malignancy prob per slice → coarse **Z-localization** (which slice drives the score) |

**Reported metrics** (per epoch, on val): **AUROC** (primary, used for best-ckpt),
**F1**, **Accuracy** (threshold 0.5).

---

## 6. Notes & current limitations

- **Classification only.** Lesion masks/boxes are preprocessed and cached but the
  loss does not use them — there is **no x/y bounding-box output** at inference
  (only `P_scan` + coarse per-slice Z-localization).
- **76 unique malignant train volumes** is the capacity ceiling; balanced batches +
  augmentation mitigate but cannot create new positives.
- **Divergences from the paper** (intentional/engineering): cancer-only positive
  (paper = biopsied benign+cancer), balanced-batch sampling (paper = weighted loss),
  `in_chans=1` (paper = 3-channel replicate), frozen stages 2–3 (paper = all layers),
  volume-level MIL (paper = slice-level labels). Architecture + 8-slice moving-average
  aggregation match the paper.
- No early stopping yet (paper used patience 5); training runs fixed 15 epochs,
  keeping the best-AUROC checkpoint.
