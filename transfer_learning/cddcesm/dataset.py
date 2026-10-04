"""
[CDD-CESM 어댑터] Categorized Digital Database for Low-energy and Subtracted CESM (Egypt, Cairo).
326 환자 / 2,006 주석. 라벨 = Radiology-manual-annotations.xlsx 의 'Pathology Classification/ Follow up'.
★ Normal 클래스 존재 → '병변 존재(lesion presence)' 태스크 지원:
   abnormal = Benign + Malignant, normal = Normal. (생검/추적 기반 pathology 라벨.)
★ 각 케이스에 DM(저에너지=표준 FFDM)과 CESM(감산=조영증강) 2종. 여기서는 **DM(FFDM) 만** 사용
   → 표준 유방촬영 lesion-presence 세트. (CESM/subtracted 는 별도 모달리티 트랙으로 후속.)
split: 공식 없음 → crc32(Patient_ID) 환자 단위 결정적 분리(누수·CC/MLO·L/R 오염 방지).
이미지: JPG (cv2). 경로: <root>/PKG - CDD-CESM/CDD-CESM/Low energy images of CDD-CESM/<Image_name>.jpg
"""
import os
import cv2
import numpy as np
import pandas as pd
from typing import List

from transfer_learning.mammo2d_base import Mammo2DDataset, deterministic_bucket

_ROOT = '/data/datasets/PKG-CDD-CESM'
_XLSX = os.path.join(_ROOT, 'Radiology-manual-annotations.xlsx')
_DM_DIR = os.path.join(_ROOT, 'PKG - CDD-CESM', 'CDD-CESM', 'Low energy images of CDD-CESM')
_LABEL_COL = 'Pathology Classification/ Follow up'


class CDDCESMDataset(Mammo2DDataset):
    name = 'cddcesm'
    task = 'lesion'   # 'lesion' = normal vs benign+malignant; 'malignancy' = benign vs malignant

    def _build_records(self) -> List[dict]:
        df = pd.read_excel(_XLSX)
        df = df[df['Type'].astype(str).str.upper() == 'DM']       # DM(FFDM) 만
        out = []
        for _, row in df.iterrows():
            name = str(row['Image_name']).strip()
            patho = str(row[_LABEL_COL]).strip().lower()
            pid = str(row['Patient_ID']).strip()
            if patho not in ('normal', 'benign', 'malignant'):
                continue
            path = os.path.join(_DM_DIR, f"{name}.jpg")
            if not os.path.exists(path):
                continue
            if deterministic_bucket(pid) != self.split:          # 환자 단위 split (두 task 공유)
                continue
            if self.task == 'malignancy':
                if patho == 'normal':
                    continue                                     # 정상 제외 → benign vs malignant
                label = 1 if patho == 'malignant' else 0         # malignant=1, benign=0
            else:
                label = 0 if patho == 'normal' else 1            # lesion presence
            out.append({'image_path': path, 'label': label, 'uid': name})
        return out

    def _compute_preproc(self, rec: dict) -> np.ndarray:
        img = cv2.imread(rec['image_path'], cv2.IMREAD_GRAYSCALE)  # [H,W] uint8
        vol = img[np.newaxis, ...]
        crop = self.preprocessor.compute_crop_rect(vol)
        return self.preprocessor.finalize_image(vol[0], crop)
