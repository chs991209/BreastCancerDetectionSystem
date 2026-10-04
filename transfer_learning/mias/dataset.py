"""
[MIAS 어댑터] mini-MIAS v1.21 — 322 PGM 유방촬영. 라벨 = 00README.pdf → mias_labels.csv.
- 이미지: DICOM 아님 → PGM(cv2)로 로드(_compute_preproc 오버라이드).
- 라벨(lesion presence): abnormal = class≠NORM(병변 존재), normal = NORM. (malignant 컬럼도 보유)
- split: L/R 쌍 = 동일 개인(patient) → 환자 단위 그룹 + class 층화 70/15/15(무누수).
경로: /data/datasets/MIAS/<image_id>.pgm , /data/datasets/MIAS/mias_labels.csv
"""
import os
import csv
import zlib
import cv2
import numpy as np
from collections import defaultdict
from typing import List

from transfer_learning.mammo2d_base import Mammo2DDataset

_ROOT = '/data/datasets/MIAS'
_CSV = os.path.join(_ROOT, 'mias_labels.csv')


def _stratified_patient_split(pat_label: dict, test_frac=0.15, val_frac=0.15) -> dict:
    """환자(patient)를 class(abnormal/normal)별로 crc32 순 정렬 후 70/15/15 → 층화·무누수."""
    by = defaultdict(list)
    for pid, lab in pat_label.items():
        by[int(lab)].append(pid)
    out = {}
    for lab, pids in by.items():
        ordered = sorted(pids, key=lambda p: zlib.crc32(str(p).encode()))
        n = len(ordered); n_test = int(round(n * test_frac)); n_val = int(round(n * val_frac))
        for i, p in enumerate(ordered):
            out[p] = 'test' if i < n_test else ('val' if i < n_test + n_val else 'train')
    return out


class MIASDataset(Mammo2DDataset):
    name = 'mias'
    task = 'lesion'   # 'lesion' = normal vs abnormal(benign+malignant); 'malignancy' = benign vs malignant

    def _build_records(self) -> List[dict]:
        rows = list(csv.DictReader(open(_CSV)))
        # 환자 split 은 항상 lesion 기준(abnormal)으로 배정 → 두 task 가 동일 환자분할 공유(무누수·비교가능)
        pat_label = defaultdict(int)
        for r in rows:
            pat_label[r['patient']] = max(pat_label[r['patient']], int(r['abnormal']))
        split_of = _stratified_patient_split(pat_label)

        recs = []
        for r in rows:
            if split_of[r['patient']] != self.split:
                continue
            path = os.path.join(_ROOT, r['image_id'] + '.pgm')
            if not os.path.exists(path):
                continue
            if self.task == 'malignancy':
                if int(r['abnormal']) == 0:
                    continue                                  # 정상 제외 → benign vs malignant
                label = int(r['malignant'])                   # 1=malignant, 0=benign
            else:
                label = int(r['abnormal'])                    # lesion presence
            recs.append({'image_path': path, 'label': label, 'uid': r['image_id']})
        return recs

    def _compute_preproc(self, rec: dict) -> np.ndarray:
        """MIAS 는 8-bit PGM → cv2 로 로드 후 동일 크롭/CLAHE/리사이즈."""
        img = cv2.imread(rec['image_path'], cv2.IMREAD_GRAYSCALE)   # [H, W] uint8
        vol = img[np.newaxis, ...]                                  # [1, H, W]
        crop = self.preprocessor.compute_crop_rect(vol)
        return self.preprocessor.finalize_image(vol[0], crop)      # [384,384] uint8
