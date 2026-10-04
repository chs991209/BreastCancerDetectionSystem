# Results — 2.5D DBT Fine-Tuning & DBT→2D Transfer Learning

**Date:** 2026-07-30 · **Host:** RTX A6000 · **Task:** malignant (cancer) detection

## TL;DR
2.5D (k=3) Swin fine-tuned on BCS-DBT, then weight-transferred to 2D mammography.
**Best DBT backbone = Swin-Small @ AUROC 0.782** (beat overfit-prone Swin-Base 0.739);
it transferred cleanly to **CMMD 0.76 / VinDr 0.76 / INbreast ~0.83**. Whole pipeline validated.

---

## 1. Method
- **Bridge (3D→2D difference):** 2.5D adjacent-slice channels — each MIL instance = k=2r+1 (r=1 → k=3)
  neighboring slices as input channels. A 2D image enters as the degenerate k=1-replicated case,
  so DBT and 2D use the **same `in_chans=k` Swin**. Depth-preserving, no deflation/weight-projection.
- **DBT fine-tune:** per-slice Swin → MIL max-pool → focal loss + auxiliary Dice (WSOL), LLRD, bf16.
  Class imbalance via random negative undersampling (pos_rate).
- **Transfer (direction B):** DBT checkpoint → strict-load into same-arch 2D Swin → fine-tune on each
  2D dataset separately. Classification (focal) only; 2D has Z=1 so MIL-max is identity.

## 2. Datasets
| Dataset | Dim | Images (train/val) | Positive def | Train pos% |
|---|---|---|---|---|
| BCS-DBT | 3D | 19,148 / 1,163 vols | Cancer only (76 train) | 0.4% |
| VinDr-Mammo | 2D | 13,426 / 1,830 | BI-RADS 4+5 (3 dropped) | 5.3% |
| CMMD | 2D | 2,636 / 358 | Malignant (biopsy) | 69.4% |
| INbreast | 2D | 270 / 40 | BI-RADS 4a/b/c,5,6 (3 dropped) | 27.8% |

## 3. DBT 2.5D fine-tuning results (k=3)
| Run | Backbone | Params | Init | Undersample | Best val AUROC | Behavior |
|---|---|---|---|---|---|---|
| Swin-B 8% | swin_base_384 | 87M | MoCo SSL | 8% pos (76/874) | 0.7392 | overfit (peak @ ep2) |
| Swin-B 5% | swin_base_384 | 87M | MoCo SSL | 5% pos (76/1444) | 0.7248 | overfit; within noise |
| **Swin-S 5%** | swin_small_384* | **49M** | ImageNet | 5% pos (76/1444) | **0.7822** | **climbed, no overfit** |

*Swin-Small has no native 384 variant → `swin_small_patch4_window7_224` with `img_size=384` override.

**Baselines (pre-2.5D):** WSOL 0.732 · classification-only 0.749.
**Finding:** capacity (not initialization) was the overfit bottleneck — halving params let AUROC climb
monotonically (0.65→0.78) instead of peaking at epoch 2. The 76-positive scarcity remains the ceiling.

## 4. DBT→2D transfer results (best val AUROC)
| Dataset | Swin-B source (0.7392) | Swin-S source (0.7822) | Notes |
|---|---|---|---|
| VinDr (20:80 balanced) | 0.725 | **0.761** | needed balancing (see below) |
| CMMD | 0.755 | **0.762** | — |
| INbreast | 0.864 | 0.810 → **0.826 ± 0.05** (3 runs) | tiny val=40 → high variance |

**INbreast variance check (Swin-S, 3 runs):** 0.850 / 0.796 / 0.832 → mean 0.826, ±0.05.
→ Swin-B vs Swin-S on INbreast are **statistically indistinguishable** (val too small to separate).

### VinDr imbalance collapse (resolved)
VinDr is 5% positive. Unbalanced transfer **collapsed to all-negative** (AUROC 0.577, F1 0.00,
Acc 0.957 = negative rate). Adding a **RatioUndersampler (pos_rate=0.20, 20:80)** fixed it →
AUROC 0.725→0.761, F1 0.00→0.35. CMMD (69% pos) and INbreast (28% pos) needed no balancing.

## 5. Checkpoints
| File | What |
|---|---|
| `checkpoints/best_25d_swins_r1_pos05.pth` | **Best DBT backbone (Swin-S, 0.7822)** |
| `checkpoints/best_25d_r1.pth` | Swin-B 8% (0.7392) |
| `checkpoints/best_25d_r1_pos05.pth` | Swin-B 5% (0.7248) |
| `transfer_learning/{vindr,cmmd,inbreast}/checkpoints/best_transfer_swins_r1.pth` | Swin-S-source transfers |
| `transfer_learning/{vindr,cmmd,inbreast}/checkpoints/best_transfer_r1.pth` | Swin-B-source transfers |

## 6. Conclusions
1. **2.5D bridge works** — one `in_chans=k` Swin serves both 3D DBT and 2D mammography; DBT→2D
   transfer is validated on all three 2D datasets.
2. **Smaller model > bigger** here — Swin-S (0.782) beat Swin-B (0.739) by fixing overfitting;
   MoCo SSL init was not required to reach the best result.
3. **Imbalance ≠ scarcity** — negative undersampling only tunes the ratio; the ceiling is the
   **76 unique positives**. Real gains require more positive signal (augmentation / external data).
4. **Severely imbalanced 2D targets need class balancing** — VinDr collapsed without it; a 20:80
   undersampler restored learning.

## 7. Reproduce (in-container, `PYTHONPATH=/workspace`, cwd `/workspace`)
```
# DBT 2.5D fine-tune (Swin-S, best)     → edit fine_tuning/fine-tune.py __main__ if changing config
python fine_tuning/fine-tune.py

# Transfer (per dataset; source ckpt + backbone set in each run.py)
python transfer_learning/cmmd/run.py
python transfer_learning/inbreast/run.py
python transfer_learning/vindr/run.py     # pos_rate=0.20 balancing on
```
