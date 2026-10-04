"""Quick m0 (ImageNet-TL) vs m1 (DBT-TL full) comparison from saved TEST predictions.
Paired bootstrap on the shared EMBED test set (correlated AUCs) → ΔAUC 95% CI + two-sided p, per seed + pooled.
실행: PYTHONPATH=/workspace python experiments/compare_m0_m1.py
"""
import json, glob, numpy as np
from sklearn.metrics import roc_auc_score

rng = np.random.RandomState(0)
DS, BT = 'embed', 'bal'
# auto-detect seeds where BOTH m0 and m1 preds exist
import os
_all = [f.split('_s')[-1][:-5] for f in glob.glob(f'experiments/preds/{DS}_{BT}_m0_2donly_s*.json')]
seeds = sorted(s for s in _all
               if os.path.exists(f'experiments/preds/{DS}_{BT}_m1_wt_llrd_s{s}.json'))


def load(meth, s):
    f = f'experiments/preds/{DS}_{BT}_{meth}_s{s}.json'
    d = json.load(open(f))
    return np.array(d['labels']), np.array(d['preds'])


def boot(y, p0, p1, n=5000):
    idx = np.arange(len(y))
    diffs = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        diffs.append(roc_auc_score(y[b], p1[b]) - roc_auc_score(y[b], p0[b]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return diffs.mean(), lo, hi, p


print(f'{DS} ({BT}) — m1 (DBT-TL) vs m0 (ImageNet-TL), paired bootstrap on shared TEST set\n')
alls = []
for s in seeds:
    y0, p0 = load('m0_2donly', s)
    y1, p1 = load('m1_wt_llrd', s)
    assert np.array_equal(y0, y1), f'label mismatch seed {s}'   # same test set
    a0, a1 = roc_auc_score(y0, p0), roc_auc_score(y1, p1)
    d, lo, hi, p = boot(y0, p0, p1)
    sig = 'sig' if (lo > 0 or hi < 0) else 'n.s.'
    print(f'  seed {s:2s}: m0={a0:.4f}  m1={a1:.4f}  Δ={a1-a0:+.4f}  [95% {lo:+.4f},{hi:+.4f}] p={p:.3f} {sig}')
    alls.append(a1 - a0)

alls = np.array(alls)
print(f'\n  across-seed: mean Δ={alls.mean():+.4f}  (per-seed Δ={[round(x,4) for x in alls]})')
