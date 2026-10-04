# Transfer-Learning Methods Beyond Weight Transfer — Plan

**Context.** The current DBT→2D transfer is **direction-B weight transfer**: load the combined DBT 2.5D Swin
checkpoint (strict) into an identical 2D Swin, then fine-tune. This document plans the **other** transfer
mechanisms — ones that transfer *knowledge* without a strict weight copy — so the paper can compare a
family of transfer strategies rather than a single method. Every method is evaluated under the objectivity
protocol in `RESEARCH_PLAN_npj.md` (5-seed, 95% CI, DeLong, TEST-once).

Baselines to beat: 2D-only (ImageNet init, no DBT knowledge) and weight-transfer full-FT (EMBED 0.775 / LLRD 0.778).

---

## Taxonomy of transfer mechanisms

| # | Method | Transfers via | Copies DBT weights? | Cost | Priority |
|---|---|---|---|---|---|
| 0 | **Weight transfer (baseline, exists)** | strict weight load + fine-tune | yes | — | ref |
| 1 | **Frozen feature extractor + linear probe** | fixed DBT features, train head only | uses (frozen) | low | P0 |
| 2 | **LP-FT** (linear-probe → fine-tune) | probe first, then unfreeze | uses | low | P0 |
| 3 | **Cross-dimensional Knowledge Distillation** | teacher soft labels (no weight copy) | **no** | medium | P1 |
| 4 | **Feature-distribution alignment (CORAL / MMD)** | match DBT vs 2D feature statistics | **no** | medium | P1 |
| 5 | **Domain-adversarial (DANN)** | gradient-reversal domain classifier | **no** | medium | P2 |
| 6 | **Intermediate-domain bridge (C-view / MIP)** | DBT→synthetic-2D data, then real 2D | **no** (data-level) | high | P1 |
| 7 | **Self-supervised pretrain on 2D (MoCo/SimCLR)** | representation learned on 2D itself | **no** | high | P2 |
| 8 | **3D→2D weight deflation** | collapse depth kernels → 2D weights | transformed | medium | P2 |
| 9 | **Adapter / LoRA (parameter-efficient)** | small trainable modules, backbone frozen | uses (frozen) | medium | P2 |

---

## Method details

### 1. Frozen feature extractor + linear probe  *(P0 — cheapest non-fine-tune transfer)*
Freeze the entire DBT backbone; train only the classifier head on 2D. Tests **how transferable the DBT
features are without any adaptation** — the classic feature-based transfer. Directly comparable to full-FT.
- Implement: `feature_extract=True` in `run_transfer` (freeze all but head). **Done in code.**

### 2. LP-FT (linear-probe then fine-tune)  *(P0)*
Phase A: freeze backbone, train head to convergence (probe). Phase B: unfreeze, full fine-tune (+LLRD).
Preserves DBT features that full-FT-from-scratch-head would distort (Kumar et al., ICLR 2022).
- Implement: `lp_ft=(probe_epochs, ft_epochs)`. **Done in code.**

### 3. Cross-dimensional Knowledge Distillation  *(P1 — already implemented, revisit)*
Frozen DBT teacher generates soft labels on the 2D image; 2D student (ImageNet init) learns
`α·KD + (1−α)·focal`. No weights copied — pure knowledge transfer. Underperformed once; revisit with
feature-level KD (match intermediate features, not just logits) and tuned α, T.
- Status: logit-KD exists in `transfer_common.kd_bce_with_logits`. **Add feature-KD variant.**

### 4. Feature-distribution alignment — CORAL / MMD  *(P1)*
Reduce the DBT→2D domain shift by aligning feature statistics: **CORAL** matches second-order stats
(covariances), **MMD** matches kernel-mean embeddings. Add an alignment loss between a batch of DBT
features (from cached DBT volumes) and 2D features. Backbone can start ImageNet or DBT-init.
- Implement: `align='coral'|'mmd'`, needs a DBT feature stream (sample cached DBT volumes per step).

### 5. Domain-adversarial (DANN)  *(P2)*
A domain classifier tries to tell DBT vs 2D features apart through a **gradient-reversal layer**; the
backbone learns domain-invariant features. Stronger but less stable than CORAL/MMD.

### 6. Intermediate-domain bridge — C-view / MIP  *(P1 — highest potential, data-level)*
Collapse DBT volumes into synthetic 2D (max/mean-intensity projection = clinical C-view). Fine-tune the 2D
head on these **DBT-labeled synthetic-2D** images first, then transfer to real 2D. Bridges the modality gap
using existing labels; no weight copy required (it's a data pipeline). See `3D_2D_전이_성능향상_방안.md`.

### 7. Self-supervised pretrain on 2D  *(P2)*
MoCo v2 / SimCLR on unlabeled 2D mammograms (EMBED is large), then supervised fine-tune. Transfers the
*method* (contrastive SSL) rather than DBT weights. Compare against the existing DBT-side MoCo SSL.

### 8. 3D→2D weight deflation  *(P2)*
From the brain-MRI reference paper: convert 3D kernels to 2D by summing/averaging over the depth axis, so a
3D-pretrained net initializes a 2D net without a same-shape strict load. Our 2.5D bridge already sidesteps
this, but deflation is a citable alternative worth a comparison row.

### 9. Adapter / LoRA  *(P2)*
Freeze the DBT backbone, insert small trainable adapters (or low-rank updates) in each Swin block.
Parameter-efficient transfer — trains <1% of params. Good for the reproducibility angle (tiny, deterministic).

---

## Evaluation matrix (all under the objectivity protocol)

Rows = methods 0–9; columns = clean datasets (EMBED, CMMD, MIAS) + malignancy set (RSNA, CBIS).
Report TEST AUROC mean ± 95% CI (5 seeds); DeLong vs weight-transfer baseline and vs 2D-only.
A method is reported as an improvement **only** if CI is non-overlapping or DeLong p<0.05 after correction.

## Suggested execution order
1. (running) LLRD ablation.
2. **P0:** methods 1 (feature-extract) + 2 (LP-FT) — cheap, multi-seed on EMBED first.
3. **P1:** method 3 (feature-KD), 4 (CORAL/MMD), 6 (C-view bridge).
4. **P2:** 5 (DANN), 7 (2D-SSL), 8 (deflation), 9 (adapters) as capacity allows.
