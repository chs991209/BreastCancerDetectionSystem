# BCS-DBT Project Dossier

**Breast Cancer DBT Detection — consolidated project record**
Updated 2026-08-25 · Host: RTX A6000 · Docker · Task: lesion presence (abnormal = benign + malignant)

> **Main purpose — reproducibility of medical-AI systems.** This project's primary goal is *reproducible,
> auditable* medical AI. The two-stage breast-cancer DBT→2D pipeline is the vehicle: every choice
> (deterministic seeding, no synthetic resampling, leakage-safe patient-grouped splits, explicit
> per-stage LLRD, disclosed label definitions) exists to make results reproducible and their limitations
> explicit. Accuracy is reported, but as reproducible evidence — not leaderboard numbers.

A two-stage lesion-presence system: 3D digital breast tomosynthesis (DBT) fine-tuning, then
2.5D-bridged transfer to 2D mammography — built for reproducibility.

Web version (theme-aware, with pipeline diagram): https://claude.ai/code/artifact/ea78cbcc-2c83-4188-a0fc-04c05544fdb6

---

## 01 · At a glance

Merging a second DBT dataset broke a long-standing single-dataset ceiling — held-out **TEST AUROC ~0.73–0.78 → 0.929**.

| Metric | Value |
|---|---|
| DBT combined — held-out **TEST** AUROC | **0.929** (VAL 0.951) |
| Abnormal training signal after merge | **76 → 1,502** (~20×) |
| 2D transfer datasets validated | 5 (DBT → mammo/US) |
| Best clean 2D transfer | MIAS 0.86; EMBED/CMMD ≈ 0.78 |

**Two-line story.** (1) **Stage 1** — a per-slice Swin-B with multiple-instance-learning (MIL) pooling
learns lesion presence on combined 3D DBT (BCS-DBT + DBT-2026). (2) **Stage 2** — those weights
transfer into an architecturally identical 2D network and fine-tune on 2D mammography, bridged by
**2.5D adjacent-slice channels** so the same network reads both 3D and 2D (a 2D image = the k=1 case).

---

## 02 · The pipeline

```
STAGE 1 · 3D DBT FINE-TUNE                    STAGE 2 · 2D TRANSFER
┌───────────────────────────────────┐        ┌──────────────────────────────────┐
│ DBT volume [Z,384,384]             │        │ 2D mammogram (k=1, degenerate)   │
│      │                             │        │      │                           │
│ 2.5D slices (k=3 ch) → Swin-B      │        │      ▼                           │
│      (per slice)                   │        │ same Swin-B  → p(lesion) 0–1     │
│      │                             │        │ (weight transfer)                │
│ MIL max over Z → scan logit        │──load─▶│                                  │
│              (AUROC .929)          │ weights│ CMMD·INbreast·MIAS·EMBED·BUSI    │
└───────────────────────────────────┘        └──────────────────────────────────┘
```

The hinge is the `load weights` arrow: Stage 1's trained weights load (strict) into an identical 2D network.

---

## 03 · Task & label definition

Authoritative in `LABEL_DEFINITIONS.md`. Task = **lesion presence**, not box-level detection.

- **abnormal (yes-lesion)** = benign-lesion + malignant · **normal** = no-lesion. Ratios reported `normal:abnormal`.
- **BCS-DBT:** abnormal = benign + cancer; normal = Normal; **Actionable is dropped** (excluded permanently).
- **DBT-2026 (Segmed):** abnormal = groups A/C/D; normal = group B, split by lesion polygon to keep it *purely* no-lesion.
- Malignant-only labelling is **RETIRED** in favour of lesion presence.

---

## 04 · Stage 1 — DBT fine-tuning

Model `DimensionAgnostic3DSwin`: per-slice Swin-B (`in_chans=k`) → MIL max-pool over Z →
moving-avg(8)→max scan aggregation, with a WSOL saliency head. Focal + gated-Dice loss,
bf16 autocast (no GradScaler).

### Combined dataset — split composition (normal : abnormal)

Combined = BCS-DBT + DBT-2026, purely-normal labels. Train uses a 9:1 undersample.

| Split | BCS-DBT | DBT-2026 | Combined | Train sampled 9:1 |
|---|---|---|---|---|
| train | 18,232 : 200 | 149 : 1,432 | 18,381 : 1,632 | 14,688 : 1,632 |
| val | 928 : 75 | 34 : 311 | 962 : 386 | — natural |
| test | 1,356 : 121 | 34 : 312 | 1,390 : 433 | — natural |

### Fine-tuning runs

| Run | Backbone | Data | VAL | TEST |
|---|---|---|---|---|
| classification-only baseline | Swin-B | BCS-DBT (malignant) | 0.749 | — |
| Phase-A WSOL baseline | Swin-B | BCS-DBT (malignant) | 0.732 | — |
| 2.5D k=3, 5% undersample | Swin-S | BCS-DBT | 0.782 | — |
| combined, stratified, stabilized | Swin-B | BCS-DBT + DBT-2026 | 0.936 | 0.906 |
| **combined, purely-normal, batch-16** | Swin-B | combined (pureN) | **0.951** | **0.929** |

**Stabilizer** (all combined runs): 3-group LLRD (5e-5 / 5e-6 / 1e-6) + 2-epoch warmup + grad-clip 0.5.
This fixed an all-negative collapse under flat LR (AUROC had cratered to ~0.51).

---

## 05 · Stage 2 — 2D transfer learning

Direction B: load the combined DBT checkpoint into an identical 2D Swin (strict load), then fine-tune
per dataset with a 1:1 balanced training set. Best AUROC per configuration:

| Dataset | full FT | freeze-shallow (train 3&4) | freeze-deep (train shallow) |
|---|---|---|---|
| MIAS | **0.864** | 0.784 | 0.705 |
| INbreast | 0.830 | 0.760 | 0.833 † |
| CMMD | **0.780** | 0.757 | 0.724 |
| EMBED | **0.775** | 0.760 | 0.753 |
| BUSI | **0.984** ‡ | 0.981 | 0.969 |

† small-n noise (INbreast val = 48). ‡ inflated (easy ultrasound task + file-level leakage).

**Ablation finding:** full fine-tune ≥ freeze-shallow > freeze-deep, everywhere. The DBT→2D shift
needs the whole network to adapt — low-level texture *and* high-level semantics. EMBED (largest,
cleanest, patient-grouped, exactly balanced) gives the clearest monotone confirmation. On Swin-B,
freeze-shallow ≈ full because 57M of 87M parameters live in stage 3 alone.

---

## 06 · Datasets & leakage integrity

| Dataset | Region | Modality | Label source | Split · leakage |
|---|---|---|---|---|
| **BCS-DBT** | Duke, USA | 3D DBT | biopsy + screening normals | ✓ patient/study |
| **DBT-2026** | Segmed, USA | 3D DBT | eCRF groups + polygons | ✓ study=patient |
| CMMD | China | 2D FFDM | **biopsy** | ✓ patient |
| EMBED | Emory, USA | 2D FFDM | **pathology** + ROI | ✓ patient |
| MIAS | UK | digitized film | severity (biopsy-informed) | ✓ patient (L/R) |
| INbreast | Portugal | 2D FFDM | BI-RADS + partial biopsy | ⚠ file-level |
| BUSI | Egypt | 2D ultrasound | biopsy-informed | ⚠ file-level |
| VinDr-Mammo | Vietnam | 2D FFDM | BI-RADS only — no biopsy | ✗ dropped |

**Leakage-safe (headline set):** EMBED · CMMD · MIAS + both DBT sources — patient-grouped, so every
view (CC/MLO, L/R) of a patient stays in one split; no view or region contamination.

**Caveated (disclose as limitations):** INbreast & BUSI ship without patient IDs → file-level split can
leak a patient's CC/MLO/L-R across train/test. INbreast val (9 abnormal) is also too small.

### Data-split integrity — per-condition verdict

| Condition | Verdict |
|---|---|
| Full FT on every dataset | ✓ (CMMD, INbreast, MIAS, EMBED, BUSI) |
| Partial (stage 3&4) on US datasets | ✓ EMBED got it (also ran on non-US as extra ablation) |
| No leakage incl. CC/MLO & L/R contamination | ✓ patient-grouped sets; ✗ INbreast & BUSI (file-level) |
| No region leakage | ✓ single-cohort per transfer; DBT patients never in any 2D test set |
| No label-accuracy imbalance | ⚠ consistent *within* a dataset; heterogeneous *across* (biopsy vs BI-RADS; lesion vs malignancy) — report per-dataset, never pooled |
| Appropriate train/test ratio + n:a info | ✓ documented; EMBED/MIAS clean; INbreast val too small |
| Multi-seed / CIs | ✗ not yet — all numbers are single-run point estimates |

---

## 07 · Reproducibility work — current thread

The paper's stated purpose is **reproducibility of the medical-AI system**. Two changes land on that.

### LLRD by stage — plain vs imbalanced

Layer-wise learning-rate decay (LLRD) builds **each Swin stage as its own optimizer param-group**.
Two profiles under test on EMBED (depth: 0=head/norm, 1=stage4, 2=stage3, 3=stage2, 4=stage1, 5=stem):

| Profile | d0 head | d1 | d2 | d3 | d4 | d5 stem |
|---|---|---|---|---|---|---|
| plain (geometric 0.75^d) | 1.00 | 0.75 | 0.56 | 0.42 | 0.32 | 0.24 |
| **imbalanced (fine-stage boost)** | 1.00 | 0.35 | 0.60 | **1.00** | **1.00** | 0.25 |

**Hypothesis behind imbalanced LLRD.** Lesions differ by *delicate, fine-scale* texture. The
high-resolution shallow stages (stage 1/2) and the decision head therefore get full learning rate,
while the coarse deep stage 4 and generic stem are held back — a deliberately non-monotone LR profile
rather than a smooth decay.

**Status: RUNNING.** Plain LLRD at epoch 5 = **AUROC 0.778** (already past the 0.775 flat-LR baseline),
still climbing; imbalanced pending. (Final numbers to be filled in on completion.)

### Removing synthetic sampling + seeding

- **No synthetic random undersampling** — the per-epoch stochastic resampler is off; the fixed train
  set is used in full every epoch (a reproducibility hole closed). Bites hardest on naturally-imbalanced
  sets (CMMD/BUSI); EMBED is 50:50 by construction, so here the gain is determinism, not rebalancing.
- **seed=42 end-to-end** — global RNG + DataLoader generator + per-worker seeding, so a run reproduces exactly.

---

## 08 · State & open threads

**Settled:** Stage 1 combined model (TEST 0.929) · 2D transfer across 5 datasets · full-vs-partial FT
ablation · leakage audit + label definition.

**Open / running:** EMBED plain-vs-imbalanced LLRD *(running)* · CDD-CESM label file *(blocked)* ·
CBIS-DDSM loader *(to build)* · multi-seed confidence intervals *(planned)* · RSNA-BCD pull *(candidate)*.

---

## 09 · Key files

| File | Role |
|---|---|
| `common/model.py` | DimensionAgnostic3DSwin — per-slice Swin-B + MIL + WSOL; `in_chans`, `backbone`, `img_size` |
| `fine_tuning_combined/fine-tune_combined.py` | combined DBT training — LLRD stabilizer, warmup+cosine, batch-16 |
| `fine_tuning_combined/combined.py` | build combined dataset (drop actionable, purely-normal), stratified split |
| `transfer_learning/transfer_common.py` | `run_transfer` — weight transfer, partial FT, LLRD (plain/imbalanced), seeding |
| `transfer_learning/<set>/` | per-dataset loader + run: cmmd · inbreast · mias · embed · busi |
| `PAPER_TABLES.md` · `LABEL_DEFINITIONS.md` | authoritative tables & label spec for the write-up |
