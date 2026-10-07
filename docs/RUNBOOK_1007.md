# RUNBOOK — how to run every stage

End-to-end reproduction order. Each run is one command; the set is grouped by phase.
Seeds used for all multi-seed runs: **{42, 1, 2, 3, 4}**.

**In-container prefix** (run from project root):
```
sg docker -c 'docker exec -w /workspace -e PYTHONPATH=/workspace bc_detection_system_a6000 python <script> [args]'
```

Dependency chain: **preprocess → DBT fine-tune (checkpoints) → 2D transfer (needs checkpoints) → aggregate**.
m0 is the only transfer condition that does **not** need a DBT checkpoint (ImageNet-only).

---

## 1. Data loading & preprocessing (once)
| script | role | output |
|---|---|---|
| `fine_tuning/prewarm_cache.py` | DICOM → preprocessed cache (BCS-DBT) | `/data/cache/preproc/...npz` + index json |

- DBT-2026 index and the 2D transfer datasets (CMMD/MIAS/CDD-CESM/CBIS/RSNA/EMBED) build/read at train time — no separate prewarm.

## 2. Training

### 2a. DBT fine-tuning — 3D (writes `checkpoints_b16/*.pth`)
| script | condition | checkpoint |
|---|---|---|
| `fine_tuning_combined/fine-tune_combined.py` | ImageNet → DBT (**m1 source**) | `combined_25d_swinb_...pth` |
| `fine_tuning_combined/train_scratch.py` | scratch → DBT (**mscratch source**) | `combined_25d_swinbscratch_...pth` |

(`fine_tuning/fine-tune.py` = legacy **solo** BCS-DBT-only pipeline; not used for the paper.)

### 2b. 2D transfer (run per seed; writes `experiments/preds/*.json`)
| script | covers |
|---|---|
| `experiments/multiseed_matrix.py <seed>` | m0 + m1, lesion, natural |
| `experiments/balanced_run.py <seed>` | lesion, balanced |
| `experiments/malignancy_matrix.py` | malignancy (loops seeds internally) |
| `experiments/malignancy_balanced_run.py <seed>` | malignancy, balanced |
| `experiments/scratch_transfer.py <seed>` | mscratch transfer |
| `experiments/scratch_balanced_lesion.py <seed>` | mscratch, balanced |
| `experiments/cosine_m1_run.py <seed> [small\|rest\|all]` | m1cos ablation (cosine head + wCE) |

## 3. Evaluation / aggregation
- TEST AUROC is computed inside each transfer run (`eval_test=True`) and saved to `experiments/preds/{name}_{arm}_{method}_s{seed}.json`.

| script | role | output |
|---|---|---|
| `experiments/organize_auc.py` | collect all preds → ordered summary | stdout / md |
| `experiments/multiseed_matrix.py summarize` | mean ± std table | `results/auroc_*.csv` |

---

## Quick full reproduction (per seed S ∈ {42,1,2,3,4})
```
# once
python fine_tuning/prewarm_cache.py
python fine_tuning_combined/fine-tune_combined.py
python fine_tuning_combined/train_scratch.py
# per seed
python experiments/multiseed_matrix.py S
python experiments/balanced_run.py S
python experiments/malignancy_balanced_run.py S
python experiments/scratch_transfer.py S
python experiments/scratch_balanced_lesion.py S
python experiments/cosine_m1_run.py S
python experiments/malignancy_matrix.py        # loops seeds itself; run once
# aggregate
python experiments/organize_auc.py
```

Note: no single top-level entry point exists yet — the transfer matrix is spread across the scripts above. A `run_all.py <seed>` could collapse Phase 2b into one command.
