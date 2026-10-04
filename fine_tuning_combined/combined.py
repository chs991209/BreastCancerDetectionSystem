"""
[BCS-DBT + DBT-2026 통합 3D 데이터셋]
두 데이터셋을 ConcatDataset 로 통합. 둘 다 (vol[k,Z,H,W], mask[1,Z,H,W], label) 규격이라
기존 collate_dbt_volumes 가 그대로 동작(가변 Z 패딩 + k채널). 기존 파일 미수정.

[라벨 정의 — yes-lesion/no-lesion]
- abnormal(yes-lesion) = benign-lesion + malignant.
- BCS-DBT: abnormal = klass ∈ {benign, cancer}; normal = klass == 'normal';
  **actionable 은 영구 제외**(생검/확정 병변 아님, ROI 박스도 benign/cancer 만 존재).
- DBT-2026: abnormal = Group ∈ {A,C,D}; normal = Group B (병변 없는 정상 스크리너).

목적: DBT-2026 병변 뷰를 BCS-DBT 에 더해 '양성 희소성' 완화 →
fine-tune → transfer-learning 파이프라인을 새 분류기로 재검증.
"""
import numpy as np
from torch.utils.data import ConcatDataset, Sampler
import torch

from fine_tuning.data_preprocessing import BCSDBTDataset
from fine_tuning_combined.dataset_dbt2026 import DBT2026Dataset

BCSDBT_ROOT = '/data/datasets/BCS-DBT'
DBT2026_ROOT = '/data/datasets/DBT-2026/DBT-2026'


class LesionBCSDBT(BCSDBTDataset):
    """[BCS-DBT 병변 라벨 오버레이] 기존 파일 미수정 위해 서브클래스로 처리.
    - actionable 볼륨은 인덱스에서 영구 제외(drop).
    - 라벨 = int(klass ∈ {benign, cancer})  (기존의 cancer-only 대신 yes-lesion).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kept = [e for e in self.index if e['klass'] != 'actionable']
        dropped = len(self.index) - len(kept)
        self.index = kept
        print(f"[LesionBCSDBT:{self.split}] dropped {dropped} actionable; "
              f"{len(self.index)} kept (abnormal=benign+cancer)")

    def __getitem__(self, idx):
        vol, mask, _ = super().__getitem__(idx)
        e = self.index[idx]
        abnormal = int(e['klass'] in ('benign', 'cancer'))
        return vol, mask, abnormal


def build_combined_dataset(split: str, cache_dir='/data/cache',
                           neighbor_slices=1, augment=False):
    """former(BCS-DBT) + new(DBT-2026) 통합 + 각 샘플의 abnormal 플래그 배열 반환."""
    bcs = LesionBCSDBT(BCSDBT_ROOT, split=split, cache_dir=cache_dir,
                       augment=augment, neighbor_slices=neighbor_slices)
    new = DBT2026Dataset(DBT2026_ROOT, split=split, cache_dir=cache_dir,
                         augment=augment, neighbor_slices=neighbor_slices)
    combined = ConcatDataset([bcs, new])
    abn_flags = np.array([int(e['klass'] in ('benign', 'cancer')) for e in bcs.index]
                         + [e['abnormal'] for e in new.index], dtype=np.int64)
    n_abn = int(abn_flags.sum())
    b_abn = sum(int(e['klass'] in ('benign', 'cancer')) for e in bcs.index)
    print(f"[Combined:{split}] {len(combined)} volumes "
          f"(BCS-DBT {len(bcs)} + DBT-2026 {len(new)}) | "
          f"abnormal(yes-lesion)={n_abn} (BCS {b_abn} + DBT2026 {n_abn - b_abn}) "
          f"| normal:abnormal = {len(combined) - n_abn}:{n_abn}")
    return combined, abn_flags


class RatioUndersampler(Sampler):
    """abnormal 전량 + normal 무작위 부분추출로 목표 abnormal 비율(pos_rate). 통합 인덱스 기준."""
    def __init__(self, abn_flags: np.ndarray, pos_rate: float = 0.20):
        self.pos_idx = np.nonzero(abn_flags == 1)[0].tolist()
        self.neg_idx = np.nonzero(abn_flags == 0)[0].tolist()
        n_pos = len(self.pos_idx)
        n_neg_keep = int(round(n_pos * (1.0 - pos_rate) / pos_rate))
        self.n_neg_keep = min(n_neg_keep, len(self.neg_idx))
        self.epoch_len = n_pos + self.n_neg_keep
        print(f"[Sampler] combined ratio-undersample: abnormal={n_pos} + normal "
              f"{self.n_neg_keep}/{len(self.neg_idx)} → epoch={self.epoch_len}, "
              f"normal:abnormal={self.n_neg_keep}:{n_pos} "
              f"(abnormal {100 * n_pos / self.epoch_len:.1f}%)")

    def __len__(self):
        return self.epoch_len

    def __iter__(self):
        perm = torch.randperm(len(self.neg_idx))[:self.n_neg_keep]
        idx = self.pos_idx + [self.neg_idx[p] for p in perm.tolist()]
        return iter([idx[o] for o in torch.randperm(len(idx)).tolist()])
