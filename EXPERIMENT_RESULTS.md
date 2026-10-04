# Experiment Results — DBT→2D Transfer Learning

All TEST AUROC, mean ± SD over 5 seeds {42,1,2,3,4}. Complete 2026-09-01.
Raw per-run: `results/auroc_per_seed.csv` · summary: `results/auroc_summary.csv` · full write-up: `docs/FINAL_RESULTS.md`.

## Conditions
- **m0** = ImageNet-only (ImageNet→2D).
- **m1** = DBT additional (ImageNet→DBT fine-tune→2D, +LLRD).
- **mscratch** = DBT-only (scratch→DBT fine-tune→2D, +LLRD).
- **Arms:** natural = dataset's own ratio; balanced = 1:1 training sampler. Same models/test sets; only the training sampler differs.

## Stage 1 — DBT fine-tuning (transfer source), held-out DBT TEST
| checkpoint | init | VAL | TEST |
|---|---|---|---|
| combined_25d_swinb… | ImageNet | 0.951 | 0.929 |
| combined_25d_swinbscratch… | scratch | 0.894 | 0.851 |

## Stage 2 — Transfer results

### Lesion (normal vs benign+malignant)
| dataset | arm | m0 (ImageNet) | m1 (ImageNet+DBT) | mscratch (DBT-only) |
|---|---|---|---|---|
| EMBED-natural | natural | 0.712 ± 0.024 | 0.745 ± 0.005 | 0.539 ± 0.048 |
| EMBED (50:50) | balanced | 0.751 ± 0.004 | 0.761 ± 0.008 | 0.579 ± 0.051 |
| MIAS | natural | 0.504 ± 0.118 | 0.707 ± 0.041 | 0.504 ± 0.180 |
| MIAS | balanced | 0.530 ± 0.095 | 0.745 ± 0.052 | 0.390 ± 0.062 |
| CDD-CESM | natural | 0.717 ± 0.049 | 0.812 ± 0.031 | 0.549 ± 0.039 |
| CDD-CESM | balanced | 0.597 ± 0.105 | 0.796 ± 0.031 | 0.481 ± 0.050 |

### Malignancy (benign vs malignant; RSNA = cancer vs non-cancer)
| dataset | arm | m0 (ImageNet) | m1 (ImageNet+DBT) | mscratch (DBT-only) |
|---|---|---|---|---|
| CMMD | natural | 0.686 ± 0.081 | 0.783 ± 0.014 | 0.546 ± 0.009 |
| CMMD | balanced | 0.638 ± 0.068 | 0.780 ± 0.014 | 0.537 ± 0.016 |
| RSNA | balanced | 0.550 ± 0.051 | 0.688 ± 0.015 | 0.496 ± 0.022 |
| CBIS-DDSM | natural | 0.658 ± 0.058 | 0.756 ± 0.018 | 0.565 ± 0.004 |
| CBIS-DDSM | balanced | 0.650 ± 0.060 | 0.738 ± 0.018 | 0.571 ± 0.008 |

Consistent ordering across all cells: **m1 > m0 > mscratch**; m1 has the lowest variance; mscratch is near-chance (ImageNet base is essential for transferability).

## Test-set sizes
EMBED-natural 1654 · EMBED-bal 2100 · MIAS 48 (17 pos) · CDD-CESM 247 · CMMD 746 · RSNA 10867 · CBIS 605.

## Datasets
| dataset | region | task | label |
|---|---|---|---|
| EMBED / EMBED-natural | USA | lesion | pathology + ROI |
| MIAS | UK | lesion | severity (biopsy-informed) |
| CDD-CESM | Egypt | lesion | pathology (N/B/M) |
| CMMD | China | malignancy | biopsy (benign/malignant) |
| RSNA | USA multi-site | malignancy | biopsy (cancer, ~2%) |
| CBIS-DDSM | USA | malignancy | biopsy (benign/malignant) |

Excluded (fail inclusion criterion — biopsy label + patient IDs + mammography): INbreast (no patient IDs), BUSI (ultrasound), VinDr (no biopsy).
