"""mscratch (DBT-only) on the BALANCED lesion datasets — completes the balanced arm.
embed (native 50:50), mias, cddcesm with balance=True. Preds: {name}_bal_mscratch_s{seed}.json
실행: PYTHONPATH=/workspace python experiments/scratch_balanced_lesion.py <seed>
"""
import sys, os
from transfer_learning.transfer_common import run_transfer
from transfer_learning.embed.dataset import EMBEDDataset
from transfer_learning.mias.dataset import MIASDataset
from transfer_learning.cddcesm.dataset import CDDCESMDataset

SCRATCH_CK = 'checkpoints_b16/combined_25d_swinbscratch_r1_pos10_strat_pureN_b16.pth'
BB = 'swin_base_patch4_window12_384'
DATASETS = {'embed': EMBEDDataset, 'mias': MIASDataset, 'cddcesm': CDDCESMDataset}
COMMON = dict(neighbor_slices=1, dbt_ckpt=SCRATCH_CK, backbone=BB, balance=True,
              epochs=10, batch_size=16, lr=1e-4, num_workers=8, llrd=True)

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    for name, ds_cls in DATASETS.items():
        preds = f"experiments/preds/{name}_bal_mscratch_s{seed}.json"
        if os.path.exists(preds):
            print(f"skip {name}/s{seed}"); continue
        print(f"\n##### {name}/mscratch/bal/s{seed} #####", flush=True)
        try:
            r = run_transfer(ds_cls, name=name, arch_tag=f'mscratch_bal_s{seed}', seed=seed,
                             eval_test=True, preds_out=preds, **COMMON)
            au = r['test_auroc'] if isinstance(r, dict) else r
            print(f"##### RESULT {name}/mscratch/bal/s{seed}: TEST AUROC={au:.4f}")
        except Exception as e:
            print(f"##### FAILED {name}/mscratch/bal/s{seed}: {type(e).__name__}: {e}")
    print(f"\n===== scratch-balanced-lesion seed {seed} COMPLETE =====")
