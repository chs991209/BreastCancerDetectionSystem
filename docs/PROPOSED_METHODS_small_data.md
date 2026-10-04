# Small / mini-dataset transfer-learning — problems & proposed methods

**Scope of the paper (explicit):** we do **not** claim to solve reproducibility. The contribution is to
(1) **characterize** the reproducibility/generalization problems honestly, and (2) **propose methods** to
address them — some validated here, some offered as directions we cannot fully solve now. This is a
legitimate, honest contribution and the right ambition for the mini-dataset transfer setting.

---

## A. Problems observed on mini / small datasets (evidence from our runs)
| Problem | Evidence in our experiments |
|---|---|
| **High variance / unstable estimates** | INbreast val n=40 (9 abnormal) → the "freeze-deep 0.833 > full 0.830" inversion is noise; MIAS test n=17 |
| **Overfitting on tiny train** | MIAS 226 train, CDD-CESM 665 train — easy to memorize |
| **File-level leakage** (no patient IDs) | INbreast, BUSI → CC/MLO/L-R can split across train/test |
| **Poor OOD generalization** | small sets can't cover the distribution → adaptation needed (E4/E7) |
| **Imbalance amplified at small n** | few positives → unstable class boundary |

---

## B. Proposed methods (tiered by how far we take them)

### B1 — Validated in this paper
- **DBT pretraining as a stronger prior** → reduces small-data overfit vs ImageNet (m1 > m0; label-efficiency E8).
- **LP-FT** (linear-probe → fine-tune) → preserves pretrained features, less distortion on small data.
- **Frozen feature-extractor (DBT-TL frozen)** → few trainable params, suited to tiny sets.
- **Multi-seed CIs + underpowered-set flagging** → never headline an n=48 result; report variance honestly.
- **Patient-grouped splits** → remove leakage wherever patient IDs exist.

### B2 — Proposed / partial (discussed; may not fully solve here)
- **Multi-source / pooled pretraining** toward a breast foundation model (the missing foundation).
- **Domain adaptation** (CORAL / MMD / DANN) for cross-dataset shift.
- **Test-time adaptation / BatchNorm recalibration** for deployment-time distribution shift.
- **Acquisition harmonization** (MammoClean-style) to reduce scanner batch-effect.
- **Few-shot / meta-learning** for very small local clinic datasets.
- **Uncertainty estimation / abstention** so small-data models fail safe.

### B3 — Acknowledged unsolved (stated as open problems)
- No true **breast foundation model** yet; full cross-region generalization remains open.
- Datasets shipped **without patient IDs** (INbreast, BUSI) cannot be retrospectively de-leaked.

---

## C. How this frames the paper
- **Claim what we validate** (B1): DBT pretraining is a better *adaptable* foundation for per-clinic
  fine-tuning, with reproducible, leakage-safe, honestly-reported numbers — and label-efficiency on small data.
- **Propose, don't overclaim** (B2): candidate remedies for the residual small-data / OOD problems.
- **Disclose openly** (B3): what remains unsolved. This honesty *is* the reproducibility contribution.
