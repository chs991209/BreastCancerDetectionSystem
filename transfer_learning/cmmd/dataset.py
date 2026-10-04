"""
[CMMD 어댑터] 5,198 단일프레임 FFDM. 라벨 = clinical xlsx 의 classification.
양성 정의: Malignant = positive, Benign = negative (BI-RADS 아님 — 생검 기반, 3 없음).
전량 abnormal(정상 없음): negative 는 '생검된 양성종양'을 의미.
라벨 조인: DICOM 태그 (PatientID, ImageLaterality) ↔ xlsx (ID1, LeftRight).
split: 공식 없음 → crc32(PatientID) 환자 단위 결정적 분리(누수 방지).
경로: <root>/cmmd/*.dcm (UUID 파일명)
"""
import os
import glob
import pydicom
import pandas as pd
from typing import List

from transfer_learning.mammo2d_base import Mammo2DDataset, deterministic_bucket

_ROOT = '/data/datasets/CMMD'
_XLSX = os.path.join(_ROOT, 'CMMD_clinicaldata_revision.xlsx')
_DCM_GLOB = os.path.join(_ROOT, 'cmmd', '*.dcm')


class CMMDDataset(Mammo2DDataset):
    name = 'cmmd'

    def _build_records(self) -> List[dict]:
        clin = pd.read_excel(_XLSX)  # cols: ID1, LeftRight, Age, number, abnormality, classification, subtype
        # (PatientID, laterality) → label
        label_map = {}
        for r in clin.itertuples(index=False):
            key = (str(r.ID1).strip(), str(r.LeftRight).strip().upper())
            label_map[key] = 1 if str(r.classification).strip().lower() == 'malignant' else 0

        recs: List[dict] = []
        for f in sorted(glob.glob(_DCM_GLOB)):
            try:
                d = pydicom.dcmread(f, stop_before_pixels=True)
            except Exception:
                continue
            pid = str(getattr(d, 'PatientID', '')).strip()
            lat = str(getattr(d, 'ImageLaterality', '')).strip().upper()
            label = label_map.get((pid, lat))
            if label is None:
                continue                                  # 라벨 없는 DICOM 스킵
            if deterministic_bucket(pid) != self.split:   # 환자 단위 split
                continue
            uid = os.path.splitext(os.path.basename(f))[0]
            recs.append({'image_path': f, 'label': label, 'uid': uid})
        return recs
