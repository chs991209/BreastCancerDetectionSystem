"""
[RSNA-BCD 어댑터] RSNA Screening Mammography Breast Cancer Detection.
54,706 단일프레임 FFDM / 11,913 환자 / 다기관(site_id). 라벨 = train.csv 의 `cancer`.
★ 태스크 = 악성(malignancy) 검출: cancer=1 → abnormal(악성), cancer=0 → normal.
   (양성종양 lesion 은 별도 라벨 없음 → 이 세트는 '병변 존재(lesion presence)'가 아니라
    '악성 검출' 트랙. RSNA/CBIS/CMMD 악성 트랙에서 사용. LABEL_DEFINITIONS 참고.)
★ 유병률 현실적: 양성 ≈ 2.1% (53,548 : 1,158) → 자연 유병률 외부검증에 적합.
split: 공식 test 라벨 없음 → crc32(patient_id) 환자 단위 결정적 분리(누수 방지, 다기관 혼재).
뷰: CC/MLO 만 사용(AT/LM/ML/LMO 소수 뷰 제외).
경로: <root>/train_images/<patient_id>/<image_id>.dcm
"""
import os
import pandas as pd
from typing import List

from transfer_learning.mammo2d_base import Mammo2DDataset, deterministic_bucket

_ROOT = '/data/datasets/RSNA/RNSA'
_CSV = os.path.join(_ROOT, 'train.csv')
_IMG_DIR = os.path.join(_ROOT, 'train_images')
_KEEP_VIEWS = {'CC', 'MLO'}


class RSNADataset(Mammo2DDataset):
    name = 'rsna'

    def _build_records(self) -> List[dict]:
        df = pd.read_csv(_CSV)
        recs: List[dict] = []
        for r in df.itertuples(index=False):
            if str(r.view).strip().upper() not in _KEEP_VIEWS:
                continue
            pid = str(r.patient_id).strip()
            if deterministic_bucket(pid) != self.split:      # 환자 단위 split
                continue
            path = os.path.join(_IMG_DIR, pid, f"{r.image_id}.dcm")
            if not os.path.exists(path):
                continue
            recs.append({'image_path': path,
                         'label': int(r.cancer),             # 1=악성, 0=정상
                         'uid': f"{pid}_{r.image_id}"})
        return recs
