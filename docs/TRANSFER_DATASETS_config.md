# Transfer-Learning Datasets — Configuration

**Purpose:** one place for every 2D transfer dataset's config — region, modality, label source/quality, task,
split ratios, balancing configs, leakage safety. Results (multi-seed TEST AUROC) are filled from
`experiments/RESULTS_multiseed.md` as the matrix completes. Ratios are **normal:abnormal** (lesion task) or
**neg:pos** (malignancy task). Last updated 2026-08-25.

---

## Roster

| Dataset | Region / origin | Modality | Label source | Task | Split unit · leakage |
|---|---|---|---|---|---|
| EMBED | Emory, USA | 2D FFDM | pathology + ROI | lesion | patient · ✅ |
| CMMD | China | 2D FFDM | biopsy | malignancy | patient · ✅ |
| MIAS | UK | digitized film | severity (biopsy-informed) | lesion | patient (L/R) · ✅ |
| CDD-CESM | Egypt (Cairo) | 2D DM (of CESM set) | pathology (Normal/Benign/Malignant) | lesion | patient · ✅ |
| RSNA-BCD | Multi-site, USA | 2D FFDM | biopsy cancer (screening) | malignancy | patient/site · ✅ |
| INbreast | Portugal | 2D FFDM | BI-RADS + partial biopsy | lesion | file-level · ⚠ leakage |
| BUSI | Egypt | 2D ultrasound | biopsy-informed | lesion | file-level · ⚠ leakage |
| CBIS-DDSM | USA | digitized film | biopsy pathology | malignancy | pending (path-map) |
| VinDr-Mammo | Vietnam | 2D FFDM | BI-RADS only | — | **dropped** (no biopsy) |

---

## Split composition & balancing configs

Ratios per split. **Balancing configs run** = which training-time class balance each dataset is run under
in the multi-seed matrix: `balanced` = 1:1 undersample; `plain` = natural ratio (no synthetic resampling).

| Dataset | train (n:a) | val | test | Natural train pos% | Balancing configs run |
|---|---|---|---|---|---|
| EMBED | 4,900 : 4,900 | 1,050 : 1,050 | 1,050 : 1,050 | 50.0% | balanced only (already 1:1) |
| CMMD | 806 : 1,830 | 106 : 252 | 198 : 548 | 69.4% (malig.) | balanced + plain |
| MIAS | 145 : 81 | 31 : 17 | 31 : 17 | 35.8% | balanced + plain |
| CDD-CESM | 218 : 447 | 30 : 61 | 93 : 154 | 67.2% | balanced + plain |
| RSNA-BCD | 37,479 : 843 | 5,353 : 126 | 10,680 : 187 | **2.20%** (realistic) | balanced + plain |
| INbreast | 195 : 75 | 31 : 9 | 61 : 16 | 27.8% | (caveated — supplement) |
| BUSI | 93 : 453 | 20 : 97 | 20 : 97 | 83.0% | (caveated — US modality) |

**Notes.**
- RSNA-BCD is the only set at realistic screening prevalence (~2%) → its `plain` config is the honest
  operating point; `balanced` is for training-signal comparison. Prime **external-validation** set.
- EMBED is a 14k balanced subset by construction; `plain` ≈ `balanced` there, so only balanced is run.
- INbreast/BUSI excluded from the headline matrix (file-level leakage); reported as caveated supplements.

---

## Methods run per dataset (multi-seed matrix)

Seeds {42, 1, 2}; every cell = held-out TEST AUROC, model selected on VAL only.

- **m0_2donly** — ImageNet init, no DBT knowledge (transfer baseline / floor).
- **m1_wt_llrd** — DBT weight transfer + full fine-tune + LLRD (current best).
- **m2_featx** — frozen DBT feature extractor + linear probe (non-fine-tune transfer).
- **m3_lpft** — linear-probe then fine-tune (LP-FT, Kumar 2022).

Additional planned transfer mechanisms (not weight transfer) in `docs/TRANSFER_METHODS_plan.md`:
feature-KD, CORAL/MMD alignment, C-view bridge, DANN, 2D-SSL, deflation, adapters.

---

## Results (filled from experiments/RESULTS_multiseed.md as jobs complete)

> _Matrix running (3 seed-parallel processes). This section is auto-summarized into
> `experiments/RESULTS_multiseed.md`; final mean ± std tables will be merged here on completion,
> with DeLong significance vs the m0 baseline and vs m1 weight-transfer._
