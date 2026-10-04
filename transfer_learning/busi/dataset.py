"""
[BUSI 어댑터] Breast Ultrasound Images (Al-Dhabyani). ⚠️ 초음파(모달리티 상이 — 유방촬영 아님).
- 이미지: PNG(회색조 초음파) → cv2 로드. _mask 파일 제외.
- 라벨(lesion presence): normal 폴더=0, benign/malignant=1(병변 존재).
- 폴더 = 클래스. 환자ID 없음 → 파일 단위 crc32 층화 split(대체로 케이스별 독립).
경로: /data/datasets/BUSI/Dataset_BUSI_with_GT/{benign,malignant,normal}/*.png
"""
import os
import glob
import zlib
import cv2
import numpy as np
from collections import defaultdict
from typing import List

from transfer_learning.mammo2d_base import Mammo2DDataset

_ROOT = '/data/datasets/BUSI/Dataset_BUSI_with_GT'


def _strat_split(items, test_frac=0.15, val_frac=0.15):
    """items: [(uid,label)]. class별 crc32 순 70/15/15 층화(파일 단위)."""
    by = defaultdict(list)
    for uid, lab in items:
        by[lab].append(uid)
    out = {}
    for lab, uids in by.items():
        ordered = sorted(uids, key=lambda u: zlib.crc32(u.encode()))
        n = len(ordered); nt = int(round(n * test_frac)); nv = int(round(n * val_frac))
        for i, u in enumerate(ordered):
            out[u] = 'test' if i < nt else ('val' if i < nt + nv else 'train')
    return out


class BUSIDataset(Mammo2DDataset):
    name = 'busi'

    def _build_records(self) -> List[dict]:
        recs_all = []
        for cls in ('benign', 'malignant', 'normal'):
            for f in glob.glob(os.path.join(_ROOT, cls, '*.png')):
                if '_mask' in os.path.basename(f):
                    continue
                label = 0 if cls == 'normal' else 1     # lesion-presence
                uid = f'{cls}_{os.path.splitext(os.path.basename(f))[0]}'
                recs_all.append((f, label, uid))
        split_of = _strat_split([(uid, lab) for _, lab, uid in recs_all])
        return [{'image_path': f, 'label': lab, 'uid': uid}
                for f, lab, uid in recs_all if split_of[uid] == self.split]

    def _compute_preproc(self, rec: dict) -> np.ndarray:
        img = cv2.imread(rec['image_path'], cv2.IMREAD_GRAYSCALE)   # [H,W] uint8
        vol = img[np.newaxis, ...]
        crop = self.preprocessor.compute_crop_rect(vol)
        return self.preprocessor.finalize_image(vol[0], crop)
