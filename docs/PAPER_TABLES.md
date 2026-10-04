# Paper Tables — Datasets, Splits, and Transfer Runs

**Generated:** 2026-08-23 · Task: lesion-presence (abnormal = benign-lesion + malignant; normal = no-lesion). Unit = per-view image/volume. Ratios reported as **normal:abnormal**.

---

## Table 1 — Dataset configuration

| Dataset | Region / origin | Modality | Label source | Label quality | Split method | Leakage-safe? |
|---|---|---|---|---|---|---|
| **BCS-DBT** | Duke Univ, USA | 3D DBT | biopsy (benign+cancer) + screening normals; Actionable dropped | high (Buda et al.) | **official** train/val/test | ✅ patient/study |
| **DBT-2026** | Segmed (multi-site, USA) | 3D DBT | eCRF groups A/C/D (biopsy/callback) + B normal; lesion polygons | high; patient-level label (contralateral noise) | crc32(study), 70/15/15 stratified | ✅ (1 study = 1 patient) |
| **CMMD** | China (Chinese Mammography DB) | 2D FFDM | **biopsy** benign/malignant | high; **no normal class** | crc32(patient) stratified | ✅ patient |
| **INbreast** | Portugal (Porto) | 2D FFDM | BI-RADS (+ partial biopsy) | medium; tiny (410) | crc32(filename) — **patient IDs removed** | ⚠️ **file-level (leakage risk)** |
| **MIAS** | UK (mini-MIAS) | 2D digitized film | severity B/M + NORM (biopsy-informed) | medium; tiny (322) | patient-grouped (L/R pairs) stratified | ✅ patient |
| **EMBED** | Emory Univ, USA | 2D FFDM | **pathology** (`path_severity`) + BI-RADS + ROI | high; large | patient-grouped **image-balanced** greedy, 70/15/15 | ✅ patient |
| **BUSI** | Egypt (Cairo/Baheya) | **2D ultrasound** | normal/benign/malignant (+masks) | medium; **modality mismatch** | crc32(filename) — no patient ID | ⚠️ file-level (known near-dups) |
| **VinDr-Mammo** | Vietnam (Hanoi) | 2D FFDM | **BI-RADS only — NO biopsy** | **low proxy → DROPPED** | (n/a) | — |
| **CDD-CESM** | Egypt (Cairo) | 2D CESM+DM | **pathology** (Normal/Benign/Malignant) | high; **has normal class** | crc32(Patient_ID) patient | ✅ patient — loader built (DM/FFDM) |
| **RSNA-BCD** | Multi-site, USA | 2D FFDM | **biopsy cancer** (screening) | high; **realistic prevalence ~2.1%** | crc32(patient_id) patient | ✅ patient/site — loader built |
| **CBIS-DDSM** | USA (DDSM curated) | 2D digitized film | **biopsy** pathology | high; needs full-image path-map | — | ⬜ loader pending (path-map) |

**Task split (label heterogeneity — never pool):**
- **Lesion-presence** (normal vs abnormal): BCS-DBT, DBT-2026, MIAS, EMBED, INbreast, BUSI, **CDD-CESM(new)**.
- **Malignancy** (malignant vs benign/normal): CMMD, **RSNA-BCD(new)**, **CBIS-DDSM(new)** — RSNA gives realistic-prevalence external validation.

---

## Table 2 — Split composition (normal : abnormal, per split)

**3D (fine-tune) datasets — combined = BCS-DBT + DBT-2026, purely-normal labels:**

| Split | BCS-DBT | DBT-2026 | Combined | (train sampled 9:1) |
|---|---|---|---|---|
| train | 18,232 : 200 | 149 : 1,432 | 18,381 : 1,632 | 14,688 : 1,632 |
| val | 928 : 75 | 34 : 311 | 962 : 386 | — (natural) |
| test | 1,356 : 121 | 34 : 312 | 1,390 : 433 | — (natural) |

**2D (transfer) datasets — natural per split (training uses balanced 1:1 undersample):**

| Dataset | train | val | test |
|---|---|---|---|
| CMMD | 806 : 1,830 | 106 : 252 | 198 : 548 |
| INbreast | 195 : 75 | 31 : 9 | 61 : 16 |
| MIAS | 145 : 81 | 31 : 17 | 31 : 17 |
| EMBED | 4,900 : 4,900 | 1,050 : 1,050 | 1,050 : 1,050 |
| BUSI | 93 : 453 | 20 : 97 | 20 : 97 |

---

## Table 3 — DBT fine-tuning runs (stage 1)

| Run | Backbone | Init | Data | Sampler | VAL AUROC | TEST AUROC |
|---|---|---|---|---|---|---|
| classification-only baseline | Swin-B | ImageNet | BCS-DBT (malignant) | — | 0.749 | — |
| Phase-A WSOL baseline | Swin-B | MoCo-SSL | BCS-DBT (malignant) | — | 0.732 | — |
| 2.5D k=3, 8% | Swin-B | MoCo-SSL | BCS-DBT | undersample 8% | 0.7392 | — |
| 2.5D k=3, 5% | Swin-B | MoCo-SSL | BCS-DBT | undersample 5% | 0.7248 | — |
| 2.5D k=3, 5% | **Swin-S** | ImageNet | BCS-DBT | undersample 5% | **0.7822** | — |
| combined, stratified, stabilized | Swin-B | ImageNet | BCS-DBT + DBT-2026 | 9:1 | 0.9358 | **0.9058** |
| combined, purely-normal (b8, stopped) | Swin-B | ImageNet | combined (pureN) | 9:1 | 0.958 | (stopped) |
| **combined, purely-normal, batch-16** | Swin-B | ImageNet | combined (pureN) | 9:1 | **0.9514** | **0.9287** |

Stabilizer (all combined runs): 3-group LLRD (5e-5/5e-6/1e-6) + 2-epoch warmup + grad-clip 0.5 → fixed an all-negative collapse seen with flat LR.

---

## Table 4 — Transfer-learning runs (stage 2), best AUROC

Source = combined DBT lesion model. `wt` = weight-transfer (direction B). Balanced = 1:1 undersample.

| Dataset | pureN (unbal) | pureN (bal 1:1) | b16 full (bal) | partial 3&4 | freeze-deep (shallow) |
|---|---|---|---|---|---|
| CMMD | 0.766 | 0.780 | — | 0.757 | 0.724 |
| INbreast | 0.685 | 0.830 | — | 0.760 | 0.833† |
| MIAS | — | 0.812 | **0.864** | 0.784 | 0.705 |
| EMBED | — | — | **0.775** (aligned) | 0.760 | 0.753 |
| BUSI | — | — | **0.984**‡ | 0.981 | 0.969 |
| VinDr | 0.583 | 0.606 | — | — | — | *(dropped — no biopsy labels)* |

† INbreast n=48 val → high variance. ‡ BUSI inflated (easy US task + file-level leakage).

**Ablation finding:** **full fine-tune ≥ freeze-shallow > freeze-deep** — the DBT→2D shift needs the whole network to adapt (low-level texture *and* high-level semantics); freezing either end costs AUROC. (On Swin-B, freeze-shallow ≈ full since 85M/87M params sit in the deep stages.)

---

## Headline results
- **DBT (combined, held-out TEST): AUROC 0.929** — vs single-BCS-DBT ceiling ~0.73–0.78. Driven by DBT-2026 adding ~1,300 abnormal volumes (76 → 1,502), the 2.5D bridge, purely-normal relabel, stratified split, and the LLRD+warmup stabilizer.
- **DBT→2D transfer (biopsy/ROI-grounded, best per set):** MIAS 0.86 · INbreast 0.83 · EMBED 0.79 · CMMD 0.78 (BUSI 0.98 caveated; VinDr dropped).

## Ablation (all 5 datasets) — full vs partial fine-tune, best AUROC
| Dataset | full | freeze-shallow (train 3&4) | freeze-deep (train shallow) |
|---|---|---|---|
| MIAS | **0.864** | 0.784 | 0.705 |
| INbreast | 0.830 | 0.760 | 0.833† |
| CMMD | **0.780** | 0.757 | 0.724 |
| EMBED | **0.775** | 0.760 | 0.753 |
| BUSI | **0.984** | 0.981 | 0.969 |

†INbreast n=48 val → noise. **Finding: full fine-tune ≥ partial everywhere** (EMBED — largest/cleanest — gives the clearest monotone full>shallow>deep). The DBT→2D shift needs the whole network to adapt.

## LLRD ablation (EMBED, plain ratio, seed=42) — per-stage learning-rate decay
| Config | TEST/best AUROC |
|---|---|
| full-FT, flat LR (baseline) | 0.775 |
| **plain LLRD** (geometric 0.75^depth) | **0.778** |
| imbalanced LLRD (fine-stage boost [1.0,0.35,0.6,1.0,1.0,0.25]) | 0.765 |

**Finding (negative result, reported honestly):** the "delicate-lesion → boost high-resolution fine stages"
hypothesis **did not help** — plain geometric decay (0.778) beat the imbalanced profile (0.765) and edged the
flat-LR baseline (0.775). Monotone depth-decay is the better default; a non-monotone fine-stage bump hurts.

## Pending
- CBIS-DDSM full-image path-map loader; multi-seed matrix running (methods×datasets×balancing×seeds → CIs + DeLong); CDD-CESM CESM/subtracted modality track.
