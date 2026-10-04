# Dataset Sample Counts (volume-level)

Task = **lesion presence** (abnormal = benign + malignant; normal = no-lesion).
Counts are per multiframe DICOM volume (one view). BCS-DBT Actionable is excluded.

## Per-source, per-split (normal / benign / malignant)

### BCS-DBT
| split | normal | benign | malignant | total |
|-------|-------:|-------:|----------:|------:|
| train | 18,232 | 124 | 76 | 18,432 |
| val   | 928 | 38 | 37 | 1,003 |
| test  | 1,356 | 61 | 60 | 1,477 |

### DBT-2026 (malignant = Group A; benign = Group C/D + B-with-polygon)
| split | normal | benign | malignant | total |
|-------|-------:|-------:|----------:|------:|
| train | 149 | 696 | 736 | 1,581 |
| val   | 34 | 157 | 154 | 345 |
| test  | 34 | 114 | 198 | 346 |

### Combined (BCS-DBT + DBT-2026)
| split | normal | benign | malignant | total |
|-------|-------:|-------:|----------:|------:|
| train | 18,381 | 820 | 812 | 20,013 |
| val   | 962 | 195 | 191 | 1,348 |
| test  | 1,390 | 175 | 258 | 1,823 |

## Down-sampled form (9:1 sampler, train only)

The `RatioUndersampler` (pos_rate=0.10) keeps **all abnormal** and randomly draws
**9 × abnormal** normals → normal:abnormal = 9:1. Count is fixed each epoch; the specific
normals re-permute per epoch. Val/test are **not** down-sampled (evaluated on the full natural set).

| split | normal | benign | malignant | total | note |
|-------|-------:|-------:|----------:|------:|------|
| train (down-sampled) | 14,688 | 820 | 812 | 16,320 | 9:1, abnormal 10.0% |
| val   | 962 | 195 | 191 | 1,348 | natural, fixed |
| test  | 1,390 | 175 | 258 | 1,823 | natural, fixed |

- Full combined total: **23,184** (20,013 / 1,348 / 1,823)
- Down-sampled combined total: **19,491** (16,320 / 1,348 / 1,823) — 3,693 normals dropped from train.
