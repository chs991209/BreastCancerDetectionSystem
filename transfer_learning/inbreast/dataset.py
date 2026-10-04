"""
[INbreast 어댑터] 410 단일프레임 FFDM. 라벨 = INbreast.csv 의 Bi-Rads.
양성 정의: BI-RADS 4a/4b/4c/5/6 = positive, 1/2 = negative, 3 = drop.
라벨 조인: csv 'File Name'(숫자 접두) ↔ AllDICOMs/<FileName>*.dcm.
split: 공식·환자ID 없음(Patient ID='removed') → crc32(File Name) 파일 단위 결정적 분리.
경로: <root>/AllDICOMs/, <root>/INbreast.csv  (';' 구분)
"""
import os
import glob
import pandas as pd
from typing import List

from transfer_learning.mammo2d_base import Mammo2DDataset, deterministic_bucket

_ROOT = '/data/datasets/INbreast/INbreast Release 1'
_CSV = os.path.join(_ROOT, 'INbreast.csv')
_DICOM_DIR = os.path.join(_ROOT, 'AllDICOMs')

_POSITIVE = {'4a', '4b', '4c', '5', '6'}
_NEGATIVE = {'1', '2'}


class INbreastDataset(Mammo2DDataset):
    name = 'inbreast'

    def _build_records(self) -> List[dict]:
        df = pd.read_csv(_CSV, sep=';', dtype=str)
        # 컬럼명 공백/케이스 견고화
        cols = {c.strip().lower(): c for c in df.columns}
        fn_col = cols['file name']
        bir_col = cols['bi-rads']

        recs: List[dict] = []
        for _, row in df.iterrows():
            fname = str(row[fn_col]).strip()
            bir = str(row[bir_col]).strip().lower()
            if bir in _NEGATIVE:
                label = 0
            elif bir in _POSITIVE:
                label = 1
            else:
                continue                                  # '3' 등 drop
            hits = glob.glob(os.path.join(_DICOM_DIR, f"{fname}*.dcm"))
            if not hits:
                continue
            if deterministic_bucket(fname) != self.split:
                continue
            recs.append({'image_path': hits[0], 'label': label, 'uid': fname})
        return recs
