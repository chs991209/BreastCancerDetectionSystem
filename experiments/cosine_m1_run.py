"""m1 + cosine head + weighted-CE ablation, across all 2D datasets and configs (m1 = best condition).
Same as m1 (DBT weight transfer, full-FT, LLRD) but head_type='cosine', loss_type='wce'.
preds: {name}_{arm}_m1cos_s{seed}.json
실행: PYTHONPATH=/workspace python experiments/cosine_m1_run.py <seed>
"""
import sys, os
from transfer_learning.transfer_common import run_transfer
from transfer_learning.embed.dataset import EMBEDDataset
from transfer_learning.embed.dataset_natural import NaturalEMBEDDataset
from transfer_learning.mias.dataset import MIASDataset
from transfer_learning.cddcesm.dataset import CDDCESMDataset
from transfer_learning.cmmd.dataset import CMMDDataset
from transfer_learning.rsna.dataset import RSNADataset
from transfer_learning.cbis.dataset import CBISDataset

CK = 'checkpoints_b16/combined_25d_swinb_r1_pos10_strat_pureN_b16.pth'
BB = 'swin_base_patch4_window12_384'
# (name, class, arm, balance) — MIAS + CDD-CESM first (report early), rest after.
COMBOS = [
    ('mias',      MIASDataset,         'plain', False),
    ('mias',      MIASDataset,         'bal',   True),
    ('cddcesm',   CDDCESMDataset,      'plain', False),
    ('cddcesm',   CDDCESMDataset,      'bal',   True),
    ('embed_nat', NaturalEMBEDDataset, 'plain', False),
    ('embed',     EMBEDDataset,        'bal',   True),
    ('cmmd',      CMMDDataset,         'plain', False),
    ('cmmd',      CMMDDataset,         'bal',   True),
    ('rsna',      RSNADataset,         'bal',   True),
    ('cbis',      CBISDataset,         'plain', False),
    ('cbis',      CBISDataset,         'bal',   True),
]
COMMON = dict(neighbor_slices=1, dbt_ckpt=CK, backbone=BB, epochs=10, batch_size=16,
              lr=1e-4, num_workers=8, llrd=True, head_type='cosine', loss_type='wce')

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    mode = sys.argv[2] if len(sys.argv) > 2 else 'all'   # 'small' = mias+cddcesm; 'rest' = others
    combos = COMBOS
    if mode == 'small':
        combos = [c for c in COMBOS if c[0] in ('mias', 'cddcesm')]
    elif mode == 'rest':
        combos = [c for c in COMBOS if c[0] not in ('mias', 'cddcesm')]
    for name, ds_cls, arm, balance in combos:
        preds = f"experiments/preds/{name}_{arm}_m1cos_s{seed}.json"
        if os.path.exists(preds):
            print(f"skip {name}/{arm}/s{seed}"); continue
        print(f"\n##### m1cos {name}/{arm}/s{seed} #####", flush=True)
        try:
            r = run_transfer(ds_cls, name=name, arch_tag=f'm1cos_{arm}_s{seed}', seed=seed,
                             balance=balance, eval_test=True, preds_out=preds, **COMMON)
            au = r['test_auroc'] if isinstance(r, dict) else r
            print(f"##### RESULT {name}/m1cos/{arm}/s{seed}: TEST AUROC={au:.4f}")
        except Exception as e:
            print(f"##### FAILED {name}/m1cos/{arm}/s{seed}: {type(e).__name__}: {e}")
    print(f"\n===== m1cos seed {seed} COMPLETE =====")
