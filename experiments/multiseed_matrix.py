"""
[다중 시드 객관 실험 매트릭스] npj Digital Medicine 재현성 프로토콜.
methods × datasets × balancing × seeds. 각 조합마다 held-out TEST(1회) + 예측 저장(DeLong용).
- 모델선택은 VAL 로만, TEST 는 최종 1회. seed 고정 → 결정적.
- 불균형 데이터셋은 balancing 축으로 {balanced(1:1 언더샘플), plain(자연비율)} 둘 다 실행.
- 재개 가능: preds json 이 있으면 스킵. 시드별 병렬 실행(argv[1]=seed) → 단일 A6000 다중 점유.
실행(단일 시드): PYTHONPATH=/workspace python experiments/multiseed_matrix.py 42
실행(요약만):     PYTHONPATH=/workspace python experiments/multiseed_matrix.py summarize
"""
import os, sys, json, itertools
import numpy as np

from transfer_learning.transfer_common import run_transfer
from transfer_learning.embed.dataset import EMBEDDataset
from transfer_learning.embed.dataset_natural import NaturalEMBEDDataset
from transfer_learning.cmmd.dataset import CMMDDataset
from transfer_learning.mias.dataset import MIASDataset
from transfer_learning.cddcesm.dataset import CDDCESMDataset
from transfer_learning.rsna.dataset import RSNADataset

CK = 'checkpoints_b16/combined_25d_swinb_r1_pos10_strat_pureN_b16.pth'
BACKBONE = 'swin_base_patch4_window12_384'
PRED_DIR = 'experiments/preds'
SEEDS = [42, 1, 2, 3, 4]   # canonical fixed seed set (N=5). 42/1/2 launched first; 3/4 chained behind.

# ── FOCUSED RUN (now): m0 vs m1 only, natural ratio, lesion datasets. ──
METHODS = ['m0_2donly', 'm1_wt_llrd']
# name → (class, task, [balance modes]). Natural ratio (plain) since none is severely imbalanced.
DATASETS = {
    'embed_nat': (NaturalEMBEDDataset, 'lesion', [False]),   # 36.5% natural (EMBED honest prevalence)
    'mias':      (MIASDataset,         'lesion', [False]),   # 35.8% natural
    'cddcesm':   (CDDCESMDataset,      'lesion', [False]),   # 67.2% natural
}
# ── DEFERRED ("all other experiments after"): restore for the full run ──
# METHODS += ['m2_featx', 'm3_lpft']            # feature-extract, LP-FT
# DATASETS['embed'] = (EMBEDDataset, 'lesion', [True])          # balanced 50:50 (balancing ablation)
# DATASETS['embed_nat'] = (NaturalEMBEDDataset,'lesion',[True,False])  # +balanced arm
# DATASETS['mias'] = (MIASDataset,'lesion',[True,False]); DATASETS['cddcesm']=(CDDCESMDataset,'lesion',[True,False])
# DATASETS['cmmd'] = (CMMDDataset,'malignancy',[True,False])    # malignancy (E2) — severe? no
# DATASETS['rsna'] = (RSNADataset,'malignancy',[True])          # ~2% → balanced (severely imbalanced)

COMMON = dict(neighbor_slices=1, backbone=BACKBONE, epochs=10,
              batch_size=16, lr=1e-4, num_workers=8)


def _pred_path(name, method, balance, seed):
    bt = 'bal' if balance else 'plain'
    return f"{PRED_DIR}/{name}_{bt}_{method}_s{seed}.json"


def run_one(name, ds_cls, method, balance, seed):
    preds = _pred_path(name, method, balance, seed)
    if os.path.exists(preds):
        with open(preds) as f:
            return json.load(f)['test_auroc']
    bt = 'bal' if balance else 'plain'
    tag = f"{method}_{bt}_s{seed}"
    kw = dict(dataset_cls=ds_cls, name=name, arch_tag=tag, seed=seed, balance=balance,
              eval_test=True, preds_out=preds, **COMMON)
    if method == 'm0_2donly':                       # 2D 단독(ImageNet) — DBT 지식 없음
        r = run_transfer(dbt_ckpt=None, **kw)
    elif method == 'm1_wt_llrd':                    # 가중치전이 + full-FT + LLRD
        r = run_transfer(dbt_ckpt=CK, llrd=True, **kw)
    elif method == 'm2_featx':                      # 동결 특징추출기 + 선형프로빙(비-미세조정 전이)
        r = run_transfer(dbt_ckpt=CK, feature_extract=True, **kw)
    elif method == 'm3_lpft':                       # LP-FT: 프로빙(4ep) → 전체 미세조정 +LLRD
        a_kw = dict(COMMON); a_kw['epochs'] = 4
        run_transfer(dataset_cls=ds_cls, name=name, arch_tag=f"lpftA_{bt}_s{seed}",
                     dbt_ckpt=CK, feature_extract=True, seed=seed, balance=balance, **a_kw)
        a_ckpt = f"transfer_learning/{name}/checkpoints/best_transfer_wt_swinlpftA_{bt}_s{seed}_r1.pth"
        r = run_transfer(init_ckpt=a_ckpt, llrd=True, **kw)
    return r['test_auroc'] if isinstance(r, dict) else r


def _combos():
    for name, (ds_cls, task, bals) in DATASETS.items():
        for method, balance in itertools.product(METHODS, bals):
            yield name, ds_cls, method, balance


def summarize():
    lines = ["# Multi-seed matrix — TEST AUROC (mean ± std across seeds)\n",
             f"Seeds={SEEDS} · TEST touched once · bal=1:1 undersample, plain=natural ratio.\n",
             "| Dataset | task | balancing | " + " | ".join(METHODS) + " |",
             "|" + "---|" * (len(METHODS) + 3)]
    for name, (ds_cls, task, bals) in DATASETS.items():
        for balance in bals:
            bt = 'balanced' if balance else 'plain'
            cells = []
            for method in METHODS:
                aus = []
                for seed in SEEDS:
                    p = _pred_path(name, method, balance, seed)
                    if os.path.exists(p):
                        with open(p) as f:
                            aus.append(json.load(f)['test_auroc'])
                if aus:
                    a = np.array(aus)
                    s = a.std(ddof=1) if len(a) > 1 else 0.0
                    cells.append(f"{a.mean():.3f}±{s:.3f}(n{len(a)})")
                else:
                    cells.append("—")
            lines.append(f"| {name} | {task} | {bt} | " + " | ".join(cells) + " |")
    out = "\n".join(lines) + "\n"
    with open('experiments/RESULTS_multiseed.md', 'w') as f:
        f.write(out)
    print(out)


if __name__ == "__main__":
    os.makedirs(PRED_DIR, exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == 'summarize':
        summarize(); sys.exit(0)
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    combos = list(_combos())
    print(f"===== seed {seed}: {len(combos)} combos =====", flush=True)
    for i, (name, ds_cls, method, balance) in enumerate(combos, 1):
        bt = 'bal' if balance else 'plain'
        print(f"\n##### [s{seed} {i}/{len(combos)}] {name}/{method}/{bt} #####", flush=True)
        try:
            au = run_one(name, ds_cls, method, balance, seed)
            print(f"##### RESULT {name}/{method}/{bt}/s{seed}: TEST AUROC={au:.4f}")
        except Exception as e:
            print(f"##### FAILED {name}/{method}/{bt}/s{seed}: {type(e).__name__}: {e}")
    print(f"\n===== seed {seed} COMPLETE =====")
