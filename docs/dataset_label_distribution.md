# BCS-DBT Label Distribution (Normal / Actionable / Benign / Malignant)

Source: official BCS-DBT label CSVs in `BCS-DBT/Metadata/`
(`BCS-DBT-labels-{train-v2, validation-PHASE-2-Jan-2024, test-PHASE-2}.csv`).
Counts are **per view** (one row = one view/volume). The four labels are mutually
exclusive (one-hot), so each row belongs to exactly one class.
"Malignant" = the `Cancer` column.

## Counts per split

| Split | Normal | Actionable | Benign | Malignant (Cancer) | Total |
|-------|-------:|-----------:|-------:|-------------------:|------:|
| Train | 18,232 | 716 | 124 | 76 | 19,148 |
| Val   | 928 | 160 | 38 | 37 | 1,163 |
| Test  | 1,356 | 244 | 61 | 60 | 1,721 |
| **All** | **20,516** | **1,120** | **223** | **173** | **22,032** |

## Ratio within each split (%)

| Split | Normal | Actionable | Benign | Malignant | Malignant ratio (neg:pos) |
|-------|-------:|-----------:|-------:|----------:|--------------------------:|
| Train | 95.22% | 3.74% | 0.65% | **0.40%** | ~252 : 1 |
| Val   | 79.79% | 13.76% | 3.27% | **3.18%** | ~31 : 1 |
| Test  | 78.79% | 14.18% | 3.54% | **3.49%** | ~28 : 1 |
| All   | 93.12% | 5.08% | 1.01% | **0.79%** | ~126 : 1 |

## Notes

- **Severe imbalance, strongest in Train** (0.40% malignant) because the official
  train split is dominated by Normal screening views, while Val/Test were enriched
  with biopsied/recalled cases (~3% malignant).
- **Positive-class definition used by the current system: `Malignant` (Cancer) only**
  — a cancer detector, not a lesion detector. This is why the imbalance is extreme
  and is handled with the balanced-batch sampler + augmentation (not relabeling).
- **Alternative ("biopsied") definition** = Benign + Malignant as positive:
  - Train: 200 (1.04%) · Val: 75 (6.45%) · Test: 121 (7.03%)
  - This is the paper's positive definition (lesion detection); **not used here**.
- **Actionable** = recalled for additional imaging but **not biopsied** → treated as
  a (hard) negative in the cancer task.
- Unique malignant volumes available for training = **76** — the key capacity ceiling.
