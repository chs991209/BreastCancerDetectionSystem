# AUROC — organized (all experiments generated so far)

TEST AUROC per (dataset, balancing, method). Multi-seed mean±sd where n>1.
Sources: experiments/preds/*.json. Provisional until all seeds land.

## Level 1 — lesion detection (default)
| dataset | balancing | method | n | seeds (TEST) | mean±sd |
|---|---|---|---|---|---|
| cddcesm | plain | ImageNet-TL (baseline) | 5 | 1:0.7112, 2:0.7229, 3:0.7802, 4:0.6427, 42:0.7278 | **0.7170**±0.0440 |
| cddcesm | plain | DBT-TL (full, default) | 5 | 1:0.7714, 2:0.8310, 3:0.8511, 4:0.7945, 42:0.8125 | **0.8121**±0.0277 |
| embed | bal | ImageNet-TL (baseline) | 3 | 1:0.7505, 2:0.7494, 42:0.7563 | **0.7520**±0.0030 |
| embed | bal | DBT-TL (full, default) | 3 | 1:0.7660, 2:0.7683, 42:0.7488 | **0.7610**±0.0087 |
| embed_nat | plain | ImageNet-TL (baseline) | 5 | 1:0.7341, 2:0.7333, 3:0.6840, 4:0.6889, 42:0.7196 | **0.7120**±0.0215 |
| embed_nat | plain | DBT-TL (full, default) | 5 | 1:0.7378, 2:0.7504, 3:0.7433, 4:0.7458, 42:0.7499 | **0.7454**±0.0046 |
| mias | plain | ImageNet-TL (baseline) | 5 | 1:0.4345, 2:0.3359, 3:0.5702, 4:0.5427, 42:0.6347 | **0.5036**±0.1059 |
| mias | plain | DBT-TL (full, default) | 5 | 1:0.7467, 2:0.7476, 3:0.7011, 4:0.6907, 42:0.6509 | **0.7074**±0.0365 |

## Level 2 — malignancy division
| dataset | balancing | method | n | seeds (TEST) | mean±sd |
|---|---|---|---|---|---|

## Current ablations (single-run, provisional)
| experiment | AUROC | note |
|---|---|---|
| EMBED LLRD plain (geometric 0.75) | 0.7779 | best LLRD |
| EMBED LLRD imbalanced (fine-stage) | 0.7651 | negative result |
| EMBED full-FT flat-LR | 0.7752 | LLRD baseline |
| Combined DBT (stage-1 fine-tune) TEST | 0.9287 | VAL 0.9514; n=1 |

