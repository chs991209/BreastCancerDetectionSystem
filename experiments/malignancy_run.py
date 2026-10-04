"""Malignancy track — m0/m1/mscratch on clean malignancy datasets.
CMMD (benign vs malignant, 30:70 → natural). RSNA (cancer vs non-cancer, ~2% → balanced, extreme imbalance).
실행(단일 시드): PYTHONPATH=/workspace python experiments/malignancy_run.py <seed>
"""
import sys, os
from transfer_learning.transfer_common import run_transfer
from transfer_learning.cmmd.dataset import CMMDDataset
from transfer_learning.rsna.dataset import RSNADataset
from transfer_learning.cbis.dataset import CBISDataset

CK_IMGNET_DBT = 'checkpoints_b16/combined_25d_swinb_r1_pos10_strat_pureN_b16.pth'
CK_SCRATCH_DBT = 'checkpoints_b16/combined_25d_swinbscratch_r1_pos10_strat_pureN_b16.pth'
BB = 'swin_base_patch4_window12_384'
# name -> (class, balance, arm_tag). CMMD natural; RSNA balanced (extreme).
DATASETS = {'cmmd': (CMMDDataset, False, 'plain'), 'rsna': (RSNADataset, True, 'bal'),
            'cbis': (CBISDataset, False, 'plain')}   # CBIS ~55:45 → natural
METHODS = {'m0_2donly': dict(dbt_ckpt=None),
           'm1_wt_llrd': dict(dbt_ckpt=CK_IMGNET_DBT, llrd=True),
           'mscratch': dict(dbt_ckpt=CK_SCRATCH_DBT, llrd=True)}
COMMON = dict(neighbor_slices=1, backbone=BB, epochs=10, batch_size=16, lr=1e-4, num_workers=8)

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    for name, (ds_cls, balance, arm) in DATASETS.items():
        for meth, mkw in METHODS.items():
            preds = f"experiments/preds/{name}_{arm}_{meth}_s{seed}.json"
            if os.path.exists(preds):
                print(f"skip {name}/{meth}/s{seed}"); continue
            print(f"\n##### {name}/{meth}/{arm}/s{seed} #####", flush=True)
            try:
                r = run_transfer(ds_cls, name=name, arch_tag=f'{meth}_{arm}_s{seed}', seed=seed,
                                 balance=balance, eval_test=True, preds_out=preds, **mkw, **COMMON)
                au = r['test_auroc'] if isinstance(r, dict) else r
                print(f"##### RESULT {name}/{meth}/{arm}/s{seed}: TEST AUROC={au:.4f}")
            except Exception as e:
                print(f"##### FAILED {name}/{meth}/{arm}/s{seed}: {type(e).__name__}: {e}")
    print(f"\n===== malignancy seed {seed} COMPLETE =====")
