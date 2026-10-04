"""m_scratch condition: transfer-learn from the SCRATCH-DBT checkpoint (random-init → DBT-finetuned).
Same protocol as m1 (full-FT + LLRD), same natural datasets/seeds. For scratch-vs-full comparison.
Preds: {dataset}_plain_mscratch_s{seed}.json  → comparable to m0_2donly / m1_wt_llrd.
실행: PYTHONPATH=/workspace python experiments/scratch_transfer.py <seed>
"""
import sys
from transfer_learning.transfer_common import run_transfer
from transfer_learning.embed.dataset_natural import NaturalEMBEDDataset
from transfer_learning.mias.dataset import MIASDataset
from transfer_learning.cddcesm.dataset import CDDCESMDataset

SCRATCH_CK = 'checkpoints_b16/combined_25d_swinbscratch_r1_pos10_strat_pureN_b16.pth'
BACKBONE = 'swin_base_patch4_window12_384'
DATASETS = {'embed_nat': NaturalEMBEDDataset, 'mias': MIASDataset, 'cddcesm': CDDCESMDataset}
COMMON = dict(neighbor_slices=1, dbt_ckpt=SCRATCH_CK, backbone=BACKBONE, balance=False,
              epochs=10, batch_size=16, lr=1e-4, num_workers=8, llrd=True)

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    import os, json
    for name, ds_cls in DATASETS.items():
        preds = f"experiments/preds/{name}_plain_mscratch_s{seed}.json"
        if os.path.exists(preds):
            print(f"skip {name} s{seed} (exists)"); continue
        print(f"\n##### mscratch {name}/s{seed} #####", flush=True)
        try:
            r = run_transfer(ds_cls, name=name, arch_tag=f'mscratch_s{seed}', seed=seed,
                             eval_test=True, preds_out=preds, **COMMON)
            au = r['test_auroc'] if isinstance(r, dict) else r
            print(f"##### RESULT {name}/mscratch/s{seed}: TEST AUROC={au:.4f}")
        except Exception as e:
            print(f"##### FAILED {name}/mscratch/s{seed}: {type(e).__name__}: {e}")
    print(f"\n===== mscratch seed {seed} COMPLETE =====")
