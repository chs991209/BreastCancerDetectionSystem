"""AUC organizer — collect every AUROC generated so far and write an ordered summary.
Sources: experiments/preds/*.json (matrix, authoritative multi-seed) + known ablations.
실행: PYTHONPATH=/workspace python experiments/organize_auc.py
"""
import json, glob, re, statistics as st
from collections import defaultdict

BALS = ('bal', 'plain')


def parse(fname):
    b = fname.split('/')[-1][:-5]                 # strip .json
    m = re.match(r'^(.*)_s(\d+)$', b)
    if not m:
        return None
    rest, seed = m.group(1), m.group(2)
    for bt in BALS:
        tok = f'_{bt}_'
        if tok in rest:
            name, method = rest.split(tok, 1)
            return name, bt, method, seed
    return None


rows = defaultdict(dict)     # (name,bt,method) -> {seed: (test,val)}
for f in sorted(glob.glob('experiments/preds/*.json')):
    p = parse(f)
    if not p:
        print('skip', f); continue
    d = json.load(open(f))
    rows[(p[0], p[1], p[2])][p[3]] = (d.get('test_auroc'), d.get('val_auroc'))

# lesion (Level 1) vs malignancy (Level 2) grouping
LESION = {'embed', 'embed_nat', 'mias', 'cddcesm'}
MAL = {'cmmd', 'cddcesm_mal', 'mias_mal', 'rsna'}
METH_ORDER = ['m0_2donly', 'm1_wt_llrd', 'm2_featx', 'm3_lpft']
# unified transfer-learning (TL) display names: all are fine-tuning a pretrained Swin on 2D.
METH_NAME = {'m0_2donly': 'ImageNet-TL (baseline)', 'm1_wt_llrd': 'DBT-TL (full, default)',
             'm2_featx': 'DBT-TL (frozen)', 'm3_lpft': 'DBT-TL (LP-FT)'}

lines = ['# AUROC — organized (all experiments generated so far)', '',
         'TEST AUROC per (dataset, balancing, method). Multi-seed mean±sd where n>1.',
         'Sources: experiments/preds/*.json. Provisional until all seeds land.', '']


def section(title, names):
    lines.append(f'## {title}')
    lines.append('| dataset | balancing | method | n | seeds (TEST) | mean±sd |')
    lines.append('|---|---|---|---|---|---|')
    for name in sorted(names):
        for bt in BALS:
            for meth in METH_ORDER:
                v = rows.get((name, bt, meth))
                if not v:
                    continue
                tests = [t for t, _ in v.values() if t is not None]
                if not tests:
                    continue
                mean = sum(tests) / len(tests)
                sd = st.pstdev(tests) if len(tests) > 1 else 0.0
                seedstr = ', '.join(f'{s}:{v[s][0]:.4f}' for s in sorted(v))
                lines.append(f'| {name} | {bt} | {METH_NAME.get(meth, meth)} | {len(tests)} | {seedstr} | **{mean:.4f}**±{sd:.4f} |')
    lines.append('')


section('Level 1 — lesion detection (default)', LESION)
section('Level 2 — malignancy division', MAL)

# known current ablations (single-run, not from preds)
lines += ['## Current ablations (single-run, provisional)',
          '| experiment | AUROC | note |', '|---|---|---|',
          '| EMBED LLRD plain (geometric 0.75) | 0.7779 | best LLRD |',
          '| EMBED LLRD imbalanced (fine-stage) | 0.7651 | negative result |',
          '| EMBED full-FT flat-LR | 0.7752 | LLRD baseline |',
          '| Combined DBT (stage-1 fine-tune) TEST | 0.9287 | VAL 0.9514; n=1 |', '']

txt = '\n'.join(lines) + '\n'
open('experiments/AUC_summary.md', 'w').write(txt)
print(txt)
print(f'total preds parsed: {sum(len(v) for v in rows.values())} across {len(rows)} configs')
