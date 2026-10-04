"""
[CBIS-DDSM 어댑터] 생검 pathology. Task = malignancy: MALIGNANT(1) vs BENIGN/BENIGN_WITHOUT_CALLBACK(0).
mass+calc train/test 4개 CSV 통합. 정상 클래스 없음(악성 트랙).
전체 영상 = `image file path`의 첫 폴더(접미사 _N 없음) 안의 단일 DICOM (ROI/crop 은 _N 폴더 → 제외).
한 영상이 여러 abnormality 행을 공유 → 폴더 접두어로 dedup, 라벨=max(malignant).
split: crc32(patient_id) 환자 단위 70/15/15 (무누수).
"""
import os, glob, zlib
import pandas as pd
from collections import defaultdict
from typing import List
from transfer_learning.mammo2d_base import Mammo2DDataset, deterministic_bucket

_ROOT = '/data/datasets/CBIS-DDSM'
_IMG_ROOT = os.path.join(_ROOT, 'CBIS-DDMS_1', 'cbis_ddsm')
_CSVS = ['mass_case_description_train_set.csv', 'mass_case_description_test_set.csv',
         'calc_case_description_train_set.csv', 'calc_case_description_test_set.csv']


class CBISDataset(Mammo2DDataset):
    name = 'cbis'

    def _build_records(self) -> List[dict]:
        rows = []
        for c in _CSVS:
            rows.append(pd.read_csv(os.path.join(_ROOT, 'Case_descriptions', c)))
        df = pd.concat(rows, ignore_index=True)

        # dedup by full-image folder prefix; label = max(malignant) over its abnormality rows
        by_img = defaultdict(lambda: {'pid': None, 'mal': 0, 'prefix': None})
        for _, r in df.iterrows():
            prefix = str(r['image file path']).split('/')[0].strip()
            if not prefix:
                continue
            mal = 1 if str(r['pathology']).strip() == 'MALIGNANT' else 0
            e = by_img[prefix]
            e['pid'] = str(r['patient_id']).strip(); e['prefix'] = prefix
            e['mal'] = max(e['mal'], mal)

        recs = []
        for prefix, e in by_img.items():
            if deterministic_bucket(e['pid']) != self.split:      # 환자 단위 split
                continue
            hits = glob.glob(os.path.join(_IMG_ROOT, prefix, '*', '*', '*.dcm'))
            if not hits:
                continue
            recs.append({'image_path': hits[0], 'label': int(e['mal']), 'uid': prefix})
        return recs
