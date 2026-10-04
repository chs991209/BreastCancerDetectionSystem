"""
[Phase B — balanced arm] m0 vs m1 with balanced (1:1) training, same lesion headline set.
Pairs with the natural-ratio run for the natural-vs-balanced comparison the user requested.
- embed (native 50:50), mias (balance=True), cddcesm (balance=True). Preds: {name}_bal_{method}_s{seed}.json
- run_one skips combos whose preds already exist (embed n=3 done → only s3/s4 run).
실행(단일 시드): PYTHONPATH=/workspace python experiments/balanced_run.py <seed>
"""
import sys
from experiments.multiseed_matrix import run_one, SEEDS
from transfer_learning.embed.dataset import EMBEDDataset
from transfer_learning.mias.dataset import MIASDataset
from transfer_learning.cddcesm.dataset import CDDCESMDataset

METHODS = ['m0_2donly', 'm1_wt_llrd']
BAL_DATASETS = {                       # balanced (1:1) counterparts of the natural set
    'embed':   EMBEDDataset,           # native 50:50 subset
    'mias':    MIASDataset,            # balance=True → 1:1 undersample
    'cddcesm': CDDCESMDataset,         # balance=True
}

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    combos = [(n, c, m) for n, c in BAL_DATASETS.items() for m in METHODS]
    print(f"===== balanced arm, seed {seed}: {len(combos)} combos =====", flush=True)
    for i, (name, ds_cls, method) in enumerate(combos, 1):
        print(f"\n##### [bal s{seed} {i}/{len(combos)}] {name}/{method} #####", flush=True)
        try:
            au = run_one(name, ds_cls, method, True, seed)   # balance=True
            print(f"##### RESULT {name}/{method}/bal/s{seed}: TEST AUROC={au:.4f}")
        except Exception as e:
            print(f"##### FAILED {name}/{method}/bal/s{seed}: {type(e).__name__}: {e}")
    print(f"\n===== balanced seed {seed} COMPLETE =====")
