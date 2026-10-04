"""Malignancy balanced arm — CMMD + CBIS with balance=True (RSNA is already balanced-only).
Runs after the natural malignancy runs. m0/m1/mscratch × 5 seeds.
실행: PYTHONPATH=/workspace python experiments/malignancy_balanced_run.py <seed>
"""
import sys, os
from transfer_learning.transfer_common import run_transfer
from transfer_learning.cmmd.dataset import CMMDDataset
from transfer_learning.cbis.dataset import CBISDataset

CK_IMGNET_DBT = 'checkpoints_b16/combined_25d_swinb_r1_pos10_strat_pureN_b16.pth'
CK_SCRATCH_DBT = 'checkpoints_b16/combined_25d_swinbscratch_r1_pos10_strat_pureN_b16.pth'
BB = 'swin_base_patch4_window12_384'
DATASETS = {'cmmd': CMMDDataset, 'cbis': CBISDataset}   # balanced arm
METHODS = {'m0_2donly': dict(dbt_ckpt=None),
           'm1_wt_llrd': dict(dbt_ckpt=CK_IMGNET_DBT, llrd=True),
           'mscratch': dict(dbt_ckpt=CK_SCRATCH_DBT, llrd=True)}
COMMON = dict(neighbor_slices=1, backbone=BB, epochs=10, batch_size=16, lr=1e-4, num_workers=8)

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    for name, ds_cls in DATASETS.items():
        for meth, mkw in METHODS.items():
            preds = f"experiments/preds/{name}_bal_{meth}_s{seed}.json"
            if os.path.exists(preds):
                print(f"skip {name}/{meth}/s{seed}"); continue
            print(f"\n##### {name}/{meth}/bal/s{seed} #####", flush=True)
            try:
                r = run_transfer(ds_cls, name=name, arch_tag=f'{meth}_bal_s{seed}', seed=seed,
                                 balance=True, eval_test=True, preds_out=preds, **mkw, **COMMON)
                au = r['test_auroc'] if isinstance(r, dict) else r
                print(f"##### RESULT {name}/{meth}/bal/s{seed}: TEST AUROC={au:.4f}")
            except Exception as e:
                print(f"##### FAILED {name}/{meth}/bal/s{seed}: {type(e).__name__}: {e}")
    print(f"\n===== malignancy-balanced seed {seed} COMPLETE =====")
