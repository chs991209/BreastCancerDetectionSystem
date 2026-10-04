"""
[VinDr-Mammo 어댑터] 20,000 단일프레임 FFDM. 라벨 = breast-level BI-RADS.
양성 정의: BI-RADS 4+5 = positive, 1+2 = negative, 3 = drop(제외).
split: 공식 training/test 사용. val 은 training 에서 crc32(study_id) 로 결정적 홀드아웃.
경로: <root>/images/<study_id>/<image_id>.dicom
"""
import os
import glob
import pandas as pd
from typing import List

from transfer_learning.mammo2d_base import Mammo2DDataset, deterministic_bucket

# 컨테이너 마운트(/data/datasets) 아래 VinDr 루트 (버전 접미사가 길어 glob 로 해석)
_ROOT_GLOB = '/data/datasets/Vindir-Mammo/vindr-mammo-*'


def _resolve_root() -> str:
    hits = sorted(glob.glob(_ROOT_GLOB))
    if not hits:
        raise FileNotFoundError(f"VinDr root not found under {_ROOT_GLOB}")
    return hits[0]


class VinDrMammoDataset(Mammo2DDataset):
    name = 'vindr'

    def _build_records(self) -> List[dict]:
        root = _resolve_root()
        df = pd.read_csv(os.path.join(root, 'breast-level_annotations.csv'))
        images_root = os.path.join(root, 'images')

        recs: List[dict] = []
        for r in df.itertuples(index=False):
            # 공식 split: test 는 그대로, train/val 은 training 을 crc32 로 재분할
            official = str(r.split).strip().lower()   # 'training' | 'test'
            if self.split == 'test':
                if official != 'test':
                    continue
            else:
                if official != 'training':
                    continue
                # training 내부에서 train/val 결정적 분리(val 약 12.5%)
                bucket = deterministic_bucket(str(r.study_id), test_frac=0.0, val_frac=0.125)
                if bucket != self.split:
                    continue

            bir = int(str(r.breast_birads).split()[-1])  # 'BI-RADS 4' → 4
            if bir == 3:
                continue                                  # drop (probably benign)
            label = 1 if bir >= 4 else 0
            path = os.path.join(images_root, str(r.study_id), f"{r.image_id}.dicom")
            if not os.path.exists(path):
                continue
            recs.append({'image_path': path, 'label': label, 'uid': str(r.image_id)})
        return recs
