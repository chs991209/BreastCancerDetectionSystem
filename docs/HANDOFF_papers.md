# Handoff — Paper-Based Context & Design Analysis

Purpose: hand a **new session** the research context we built around three papers, the
conclusions reached, and the open design questions — so it can reason about the
BCS-DBT system without re-deriving everything. Pair with `CLAUDE.md` (repo/ops) and
`PROJECT_STATUS.md` (build state).

---

## 1. The three papers

### A. s41598-024-72707-2 — Kassis et al., *Sci Rep 2024* (`s41598-024-72707-2.pdf`)
"Detection of breast cancer in DBT with vision transformers."
- **Task = CLASSIFICATION** (tumor-presence), *despite "Detection" in the title* (clinical sense, not object detection). Output = a probability; **no boxes**.
- Backbone: **Swin-B @384** (beats ResNet101 & ViT). ImageNet-21k, grayscale→3ch, all layers fine-tuned.
- Per-slice `p_z = σ(FC(feat))` → **scan = max_z( moving-avg₈(p_z) )** → **case = ½(P_CC + P_MLO)**.
- Weighted BCE; AdamW 1e-4 cosine+warmup; batch 32; early stop patience 5.
- **Positive = biopsied benign + cancer** (~1:1.5 slice balance). Own Soroka dataset (3,831 train / 685 test scans). Case AUC **0.934** @384.

### B. 2403.13148 — Du, Hooley, Lewin, Dvornek (Yale), *ISBI 2024* (SIFT-DBT) (`2403.13148v1.pdf`, code `github.com/XYPB/SIFT_DBT`)
"Self-supervised Initialization and Fine-Tuning for Imbalanced DBT Classification."
- **Task = CLASSIFICATION** (abnormal slice/volume). **CNN backbone (ResNet50 / 7-layer)** — *not* a transformer.
- **Stage 1 SSL — MoCo v2** on 2D slices: `ℒ=−log[exp(x·x₊/τ)/Σexp(x·xᵢ/τ)]`, `θ′←mθ′+(1−m)θ`; τ=0.2, queue S=4096, m=0.999 (MoCo-v2 defaults). **DBT positive pairs**: other-view slice (p=0.5) else neighbor within ±k, **k=9**. Weakened aug (`cj-strength 0.2`). Pretrain 4000 ep, SGD 0.015.
- **Stage 2 fine-tune**: patch-level (448², abnormal patch must contain tumor), **discriminative LR `lrℓ=base_lr/η^ℓ`, η=2.8** (repo `--lr-decay`), balanced ~1:1 batch, CE, 50 ep.
- **Stage 3 aggregation**: `f_a = (1/N)Σ pⱼ` (N=20 patches) → slice; **volume = max_z(slice)**; threshold minimizes normal/abnormal recall gap.
- **Positive = abnormal = benign+cancer**. BCS-DBT re-split 7:1:2 (train **139** abnormal volumes). Volume AUC **92.69%** (ResNet50).

### C. 3D Self-Supervised… Brain Tumor MRI — Halder & Babli, *ICECTE 2026* (`3D_Self-Supervised…pdf`)
"3D SSL Pre-Training and **3D-to-2D Transfer** (deflation)."
- **Stage 1**: 3D **MAE** on unlabeled BraTS volumes (4×128³ → 16³ patches → 512 tokens, mask r=0.75 → 3D ViT enc 6L/8H/384 → decoder → MSE on masked). `ℒ_MAE=(1/|M|P)Σ‖X−X̂‖²`.
- **Stage 2 — DEFLATION** (deterministic): patch-embed **depth-average** `W2D=(1/Pz)Σ_d W3D` (Eq 2), modality→RGB avg (Eq 3), **copy transformer blocks**, **interpolate** pos-emb → 2D ViT init.
- **Stage 3**: 2D fine-tune (ViT-S, 6 blocks, LLRD, + auxiliary distillation from frozen deflated teacher first 20 ep). Deflated init wins **low-label** (10%: +0.125 F1 vs DINO); ImageNet wins at 100%. Masking rule: r=0.75 for small unlabeled sets, r=0.50 for larger.

---

## 2. Key conclusions we reached (do not re-derive)

1. **Classification vs detection.** *Both* s41598 and SIFT-DBT are **classification** (probability, no boxes). The **BCS-DBT challenge** (Buda et al.) is the actual *detection* benchmark (boxes, FROC). Our project's **Dice-"WSOL" is a weak localization add-on, NOT a detector** (single thresholded box from a saliency map, sparse supervision, no FROC/mAP objective, box in 384-padded space). Calling it "WSOL" is a mild misnomer — it uses GT box masks (Dice), so it's really an auxiliary mask-alignment loss.

2. **Deflation is mathematically valid but lossy — with a *critical* failure mode.**
   - `W2D=(1/Pz)Σ_d W3D` is a projection onto the **depth-mean**; the `1/Pz` scaling is self-consistent (LayerNorm absorbs residual scale/bias), so **not a hard error**.
   - **Critical risk: dead channels.** Any 3D filter with a **zero-mean depth profile** (through-plane derivative/edge — exactly what 3D SSL learns) → `W2D≈0` → dead embedding channel; LayerNorm can't revive it; irreversible at the input layer.
   - **Purpose-defeating for DBT:** depth-averaging discards the through-plane lesion-continuity signal — the very reason to go 3D.

3. **Transfer moves knowledge, not data.** No transfer/deflation/distillation can add information absent from the 2D input.
   - For a **DBT slice**, depth info exists in *other slices* → recoverable by **2.5D / 3D input / aggregation**, NOT by transfer. Deflation *discards* it.
   - For a **single 2D mammogram**, depth was never acquired → gone; transfer only improves *priors / label-efficiency*.

4. **Bridging 3D-Swin vs 2D-model knowledge:** split into (a) representational gap → **knowledge distillation** solves it (no projection loss, beats deflation); (b) input-info gap → only **2.5D/3D input** solves it. Full fix = **KD (3D teacher → 2D Swin student) + 2.5D input**.

5. **Imbalance ≠ scarcity (the central limiter).** Balancing the ratio (samplers/focal) removes the all-negative collapse but does **not** add positives. The system is capped by **76 unique malignant volumes** → overfitting/weak generalization regardless of balance. Confirmed empirically (balanced runs plateaued ~0.73; SSL ceiling-break underperformed at 0.70). **The only real ceiling-lifters are more positive signal** (EMBED ~1,585 malignant; strong augmentation; or relabel to biopsied=200) and preserving resolution (patches, 2.5D).

6. **"SIFT-on-Swin" ≈ our current system.** SIFT-DBT's method on a Swin backbone = Phase-B MoCo-on-Swin (already built) + SIFT patch-MIL + s41598 aggregation. It is **deflation-free** (2D-native SSL), so it avoids §2's flaws — but still hits the 76-positive ceiling. Patch-size clash to resolve: SIFT 448² vs Swin `window12_384` → use 384² patches.

---

## 3. Open design questions for the new session
- **Target deliverable: classification (scan/case probability) or box-level detection?** Decides whether the Dice head stays a nice-to-have or is replaced by a real detector (FROC-scored).
- **Positive definition: keep Cancer-only (76, current) or move to biopsied benign+cancer (200)?** The latter is easier/less imbalanced (what both papers use) but changes the clinical task to *lesion* detection.
- **Break the ceiling: EMBED 2D pretraining** (tables present at `EMBED/tables/`, images NOT downloaded) vs stronger augmentation vs KD+2.5D. All orthogonal to sampling.
- **Deflation: skip it** (recommended — lossy, self-defeating for DBT). If 3D wanted, prefer **KD + 2.5D**.

## 4. Diagram artifacts produced (string diagrams, claude.ai/code/artifact/…)
- s41598 architecture `2346ad01-…`; four processes `f750c368-…`; scan-vs-case `6c9ae534-…`;
  scan-based formulas `eb7e0d4b-…`; case-based formulas `f8c8bac6-…`; scan-based-in-ML `a5f2da41-…`.
- BCS-DBT dataset structure `05ddc288-…`; slice-level annotation `8c1a4f60-…`.
- SIFT-DBT full pipeline `c8d002a3-…`; positive pairs `4d2d4860-…`; aug+dimensions `fb465c20-…`; formulas `845679eb-…`.
- 3D→2D deflation `c4f203d7-…`; SIFT-on-Swin synthesis `30d5af8c-…`.
(Artifacts are private to the owner; re-open via claude.ai. They are illustrations — the code of record is the `.py` files + `PROJECT_STATUS.md`.)
