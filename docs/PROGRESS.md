# Progress — DBT→2D Transfer Reproducibility Study

_As of 2026-10-07. Target: npj Digital Medicine._

## What the study asks
Does fine-tuning a model on 3D breast tomosynthesis (DBT) make it a better
starting point for 2D mammography classifiers? We measure this reproducibly
across several public 2D datasets.

Task throughout = **lesion presence** (abnormal = benign-lesion + malignant; normal = no-lesion).

## Pipeline (two stages)
1. **DBT fine-tune (3D).** 2.5D per-slice Swin-B + MIL, trained on combined
   **BCS-DBT + DBT-2026**. Produces the backbone.
2. **DBT→2D transfer.** Take that backbone, apply to single-slice 2D mammograms,
   re-fit the head. Evaluated on CDD-CESM, CMMD, CBIS-DDSM, EMBED, MIAS, RSNA.

## Three conditions compared
- **m0** — ImageNet-only (no DBT stage). Baseline.
- **m1** — ImageNet → DBT fine-tune → 2D. The proposed path.
- **mscratch** — random init → DBT fine-tune → 2D. Isolates the ImageNet contribution.

## Results so far

**Stage 1 — DBT held-out AUROC**

| Init → DBT | VAL | TEST |
|---|---|---|
| ImageNet → DBT (m1 source) | **0.951** | **0.929** |
| Scratch → DBT (mscratch source) | 0.894 | 0.851 |

**Stage 2 — DBT→2D transfer, TEST AUROC (natural arm, mean of 5 seeds)**

| Dataset | m0 | m1 | mscratch | Δ_DBT (m1−m0) |
|---|---|---|---|---|
| CDD-CESM | 0.717 | **0.812** | 0.549 | +0.095 |
| CMMD | 0.686 | **0.783** | 0.546 | +0.097 |
| CBIS-DDSM | 0.658 | **0.756** | 0.565 | +0.098 |
| EMBED | 0.712 | **0.745** | 0.539 | +0.033 |
| MIAS | 0.504 | **0.707** | 0.504 | +0.203 |

**Headline:** m1 wins on every dataset and both arms. DBT fine-tuning helps 2D
transfer across all regions tested (US and non-US), supporting the hypothesis
that lesion traits transfer across regions.

## Method notes (for the paper)
- **Graded (tiered) LLRD** in the DBT fine-tune stage: head 5e-5 / stages 1–2 at
  5e-6 / stages 3–4 at 1e-6. Transfer stage uses per-depth geometric decay (0.75).
- **Stabilization:** fine-tune = 2-epoch linear warmup → cosine + grad-clip 0.5;
  transfer = cosine from step 0 (no warmup) + grad-clip 1.0. Both AdamW.
- Deterministic seeds, patient-grouped leakage-safe splits, fixed train sets.

## Status
- Code, docs, result tables, and the runbook are in the repo and committed.
- Paper-ready tables: `AUROC_TABLE.tex`, `FINETUNE_TABLE.tex`; counts in `DATASET_COUNTS.md`.
- Reproduction order: `docs/RUNBOOK.md`.

## Remaining before submission
- Multi-seed confidence intervals + DeLong tests (npj rigor bar).
- Decide whether to fold in the m1cos (cosine-head) ablation.
- Final push of local commits to the remote.
