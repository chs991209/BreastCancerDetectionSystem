"""
[E2 — Level-2 malignancy division 전이 매트릭스] benign vs malignant.
lesion 계층의 하위 분할(malignant ⊂ lesion). 정상 제외, benign(0) vs malignant(1).
데이터셋: CMMD(이미 malignancy), CDD-CESM(task='malignancy'), MIAS(task='malignancy').
methods m0–m3 × balancing × seeds. preds 명명은 메인과 동일 규약 → 통합 요약 가능.
실행: PYTHONPATH=/workspace python experiments/malignancy_matrix.py
전처리 캐시는 각 base 데이터셋('cmmd'/'cddcesm'/'mias')과 공유(같은 uid).
"""
import os
from experiments.multiseed_matrix import run_one, SEEDS, METHODS
from transfer_learning.cmmd.dataset import CMMDDataset
from transfer_learning.cddcesm.dataset import CDDCESMDataset
from transfer_learning.mias.dataset import MIASDataset


class CDDCESMMalignancy(CDDCESMDataset):
    name = 'cddcesm_mal'
    task = 'malignancy'
    def _preproc_cache_path(self, rec):
        return os.path.join(self.cache_dir, 'cddcesm', 'preproc', self.split, f"{rec['uid']}.npz")


class MIASMalignancy(MIASDataset):
    name = 'mias_mal'
    task = 'malignancy'
    def _preproc_cache_path(self, rec):
        return os.path.join(self.cache_dir, 'mias', 'preproc', self.split, f"{rec['uid']}.npz")


# name → (class, [balance modes]). CMMD/CDD-CESM imbalanced → both; MIAS-mal tiny → balanced only.
MAL_DATASETS = {
    'cmmd':        (CMMDDataset,      [True, False]),
    'cddcesm_mal': (CDDCESMMalignancy, [True, False]),
    'mias_mal':    (MIASMalignancy,   [True]),
}

if __name__ == "__main__":
    combos = [(n, c, m, b) for n, (c, bals) in MAL_DATASETS.items()
              for m in METHODS for b in bals]
    total = len(combos) * len(SEEDS)
    i = 0
    for seed in SEEDS:
        for name, ds_cls, method, balance in combos:
            i += 1
            bt = 'bal' if balance else 'plain'
            print(f"\n##### [malignancy {i}/{total}] {name}/{method}/{bt}/s{seed} #####", flush=True)
            try:
                au = run_one(name, ds_cls, method, balance, seed)
                print(f"##### RESULT {name}/{method}/{bt}/s{seed}: TEST AUROC={au:.4f}")
            except Exception as e:
                print(f"##### FAILED {name}/{method}/{bt}/s{seed}: {type(e).__name__}: {e}")
    print("\n===== malignancy matrix COMPLETE =====")
