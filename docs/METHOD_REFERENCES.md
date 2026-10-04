# Method references

Citations for the learning methods used, mapped to where they appear in the code.
Verify each before final submission (I have not re-fetched every DOI; some attributions are conventional/multiple).

## Architecture / representation
| Method | Where | Reference |
|---|---|---|
| Swin Transformer (swin_base_patch4_window12_384) | `common/model.py` | Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows," ICCV 2021. |
| 2.5D adjacent-slice channels | `fine_tuning/data_preprocessing.py` | Roth et al., "A New 2.5D Representation for Lymph Node Detection...," MICCAI 2014. |
| Multiple-instance learning, max-pool over slices | `common/model.py`, `max(dim=1)` | Ilse et al., "Attention-based Deep Multiple Instance Learning," ICML 2018 (max-pool is the classic MIL pooling; attention variant here for reference). |
| WSOL saliency (CAM-style head) | `common/model.py` | Zhou et al., "Learning Deep Features for Discriminative Localization," CVPR 2016. |
| Moving-average(8)→max scan aggregation | `fine-tune.py` | Method from the DBT paper `papers/s41598-024-72707-2.pdf` (Swin classification + moving-avg→max→CC/MLO aggregation). |

## Optimization / training
| Method | Where | Reference |
|---|---|---|
| AdamW (decoupled weight decay) | transfer/fine-tune | Loshchilov & Hutter, "Decoupled Weight Decay Regularization," ICLR 2019. |
| Cosine annealing LR (SGDR) | `CosineAnnealingLR` | Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts," ICLR 2017. |
| LR warmup (linear) | `LinearLR`, warmup_epochs | **Representative: Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour," 2017** (linear warmup, large-batch stability). Transformer-specific: Vaswani et al., "Attention Is All You Need," NeurIPS 2017. |
| Layer-wise LR decay (LLRD) / discriminative fine-tuning | `build_llrd_groups` | **Representative: Howard & Ruder, "Universal Language Model Fine-tuning (ULMFiT)," ACL 2018** (origin). Modern "LLRD" name: Clark et al., "ELECTRA," ICLR 2020. |
| Warmup as stabilization (early-training adaptive-LR variance) | stabilizer | **Liu et al., "On the Variance of the Adaptive Learning Rate and Beyond (RAdam)," ICLR 2020** — explains why warmup stabilizes training. |
| Gradient clipping | `clip_grad_norm_` | Pascanu et al., "On the difficulty of training recurrent neural networks," ICML 2013. |
| Mixed-precision (bf16 autocast) | training loops | Micikevicius et al., "Mixed Precision Training," ICLR 2018. |
| Focal loss | `focal_loss_with_logits` | Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017. |

Note: the "stabilizer" (3-group LLRD + warmup→cosine + grad-clip) is our combination; cite the components — Howard & Ruder (LLRD), Goyal (warmup), Liu/RAdam (why warmup stabilizes), Pascanu (clipping). No single source covers the combination.

## Transfer learning
| Method | Where | Reference |
|---|---|---|
| ImageNet pretraining as transfer base | m0/m1 | Deng et al., ImageNet, CVPR 2009; Kornblith et al., "Do Better ImageNet Models Transfer Better?," CVPR 2019. |
| LP-FT (linear-probe then fine-tune) | `transfer_common.py` (feature_extract/init_ckpt) | Kumar et al., "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution," ICLR 2022. |
| 3D→2D transfer (deflation, reference approach) | `papers/3D_Self-Supervised...Brain_Tumor...pdf` | 3D MAE → 3D→2D deflation → 2D ViT (brain MRI). |
| MoCo v2 SSL (reference; SIFT-DBT) | `papers/2403.13148v1.pdf` | Chen et al., "Improved Baselines with Momentum Contrastive Learning (MoCo v2)," 2020; SIFT-DBT repo XYPB/SIFT_DBT. |

## Evaluation / statistics
| Method | Where | Reference |
|---|---|---|
| AUROC comparison (DeLong) | analysis | DeLong et al., "Comparing the Areas under Two or More Correlated ROC Curves," Biometrics 1988. |
| Reporting standards | plan | TRIPOD+AI (Collins et al., 2024); CLAIM (Mongan et al., Radiology: AI 2020). |

## Reproducibility / motivation (context)
| Topic | Reference |
|---|---|
| AI reproducibility (transparency, code/data) | Haibe-Kains et al., "Transparency and reproducibility in artificial intelligence," Nature 2020. |
| Dataset-origin shortcuts, cross-dataset pooling degrades screening | Hajishafiezahramini et al., "Dataset-Origin Signatures and Shortcut Learning in Screening Mammography AI," arXiv:2607.15416, 2026. |
| Shortcuts in screening mammography DL | "Problems and shortcuts in deep learning for screening mammography," arXiv:2303.16417, 2023. |

Reference PDFs on disk: `papers/`.
