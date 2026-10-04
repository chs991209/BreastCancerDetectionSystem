# Experiment Audit — current transfer-learning (lesion-presence default)

**Scope (2026-08-25):** default task = **lesion presence, normal : abnormal (abnormal = benign + malignant)**;
default method = **general transfer (weight-transfer + full fine-tune)**. Former/historical experiments are
out of scope. This doc covers only the *current* transfer-learning matrix and its statistics.

Matrix status: **3/108 combos done** — no cell has final multi-seed statistics yet.

---

## A. Transfer datasets — task fit & validity (current scope only)

| Dataset | Region | Supports lesion-presence? | Role | Validity flag |
|---|---|---|---|---|
| **EMBED** | USA | ✅ (normal vs ROI) | **default** | balanced 50:50 → optimistic; use **EMBED-natural** for honest prevalence |
| **EMBED-natural** | USA | ✅ | **default** (honest prevalence 36.5%) | random-sampled, fixed seed, same test for all methods |
| **MIAS** | UK | ✅ (NORM vs B/M) | **default** | small (n≈322) — report but low power |
| **CDD-CESM** | Egypt | ✅ (Normal/Benign/Malignant) | **default** | patient-grouped, pathology — clean |
| CMMD | China | ❌ no normal (benign vs malignant) | secondary (**malignancy** track) | not lesion-presence; report separately |
| RSNA-BCD | USA multi-site | ❌ cancer vs non-cancer | secondary (**malignancy** track, external/prevalence anchor) | not lesion-presence |
| INbreast | Portugal | ✅ but ⚠ | excluded from headline | **file-level leakage** + val n=40 (9 abn) |
| BUSI | Egypt | ✅ but ✗ | excluded | **ultrasound** (modality mismatch) + file-level leakage |
| VinDr | Vietnam | — | dropped | BI-RADS only, no biopsy |

**Default lesion-presence transfer set = EMBED · EMBED-natural · MIAS · CDD-CESM.**
Regional coverage within the default: USA (EMBED), UK (MIAS), Egypt (CDD-CESM).
Malignancy (CMMD, RSNA) is a clearly-separated secondary track, never pooled with lesion results.

---

## B. Managed paper tables (proper statistics)

Every cell = **mean ± 95% CI over seeds, n, DeLong p vs baseline** (predictions saved in
`experiments/preds/*.json`). Populated as the matrix completes.

### T1 — Default: lesion-presence 2D transfer-learning (variants × datasets)
All arms are 2D transfer-learning (fine-tune pretrained Swin on 2D); differ by source × strategy:
**m0 ImageNet-TL (baseline)** · **m1 DBT-TL full-FT (default)** · **m2 DBT-TL frozen** · **m3 DBT-TL LP-FT**.

| Dataset | TL variant | n | TEST AUROC (mean ± 95% CI) | DeLong vs ImageNet-TL |
|---|---|---|---|---|
| EMBED | m0 ImageNet-TL | 3 | 0.752 ± 0.004 | (baseline) |
| EMBED | m1 DBT-TL (full) | 3 | **0.761 ± 0.011** | *pending* |
| EMBED | m2 DBT-TL (frozen) | – | *pending* | – |
| EMBED | m3 DBT-TL (LP-FT) | – | *pending* | – |
| EMBED-natural · MIAS · CDD-CESM × m0–m3 | | – | *pending* | – |

### T2 — Balancing config (balanced vs plain/natural), default lesion datasets
| Dataset | balanced (mean±CI) | plain/natural (mean±CI) |
|---|---|---|
| EMBED-natural · MIAS · CDD-CESM | *pending* | *pending* |

### T3 — Cross-region generalization (lesion-presence; train region → test region)
| Train ↓ / Test → | USA (EMBED) | UK (MIAS) | Egypt (CDD-CESM) |
|---|---|---|---|
| USA / UK / Egypt | *pending (off-diagonal = regional gap)* | | |

### T-sec — Secondary malignancy track (separate table, never pooled)
| Dataset | Method | n | TEST AUROC (mean ± 95% CI) |
|---|---|---|---|
| CMMD · RSNA × m0–m3 | | – | *pending* |

**Stats rules:** ≥3 seeds (5 target), 95% CI (t across seeds + bootstrap on test), DeLong pairwise with
Holm–Bonferroni, ECE calibration on finals, TEST touched once. No single-seed number reported without an
"n=1, provisional" tag.
