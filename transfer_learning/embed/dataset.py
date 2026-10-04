"""
[EMBED 어댑터] Emory EMBED 2D FFDM (다운로드된 균형 서브셋 14k).
- 이미지: DICOM → base 의 read_volume 로 로드(오버라이드 불필요).
- 라벨(lesion presence): abnormal = num_ROI>0(병변 ROI 존재), normal = asses=='N' & num_ROI==0.
  (서브셋 생성 시와 동일 정의; B/P/A 등 모호 케이스 제외, 2D only.)
- 디스크에 실제 존재하는 파일만 인덱싱(= 우리가 받은 7k+7k 서브셋).
- split: 환자(empi_anon) 단위 + class 층화 70/15/15(무누수).
경로: 테이블 /workspace/EMBED/tables/*.csv, 이미지 /data/datasets/EMBED/<anon_dicom_path 에서 images/ 제거>
"""
import os
import zlib
import pandas as pd
from collections import defaultdict
from typing import List

from transfer_learning.mammo2d_base import Mammo2DDataset

_TABLES = '/workspace/EMBED/tables'
_IMG_ROOT = '/data/datasets/EMBED'


def _balanced_group_split(pat_counts: dict, fracs=(('train', 0.70), ('val', 0.15), ('test', 0.15))) -> dict:
    """[환자 그룹 + 이미지 레벨 균형 split] 각 환자(무누수)를 통째로 배정하되,
    train/val/test 의 normal/abnormal '이미지 수'가 목표 비율에 최대한 맞도록 그리디 배정.
    → 모든 split 이 전체(≈1:1)와 같은 normal:abnormal 로 정렬(deterministic: 큰 환자부터, crc32 tiebreak).
    pat_counts: {pid: [n_normal, n_abnormal]}."""
    tot = [sum(c[k] for c in pat_counts.values()) for k in (0, 1)]
    target = {sp: [f * tot[0], f * tot[1]] for sp, f in fracs}
    cur = {sp: [0.0, 0.0] for sp, _ in fracs}
    order = sorted(pat_counts, key=lambda p: (-(sum(pat_counts[p])), zlib.crc32(str(p).encode())))
    out = {}
    for pid in order:
        nn, na = pat_counts[pid]
        best, best_score = None, None
        for sp, _ in fracs:
            # resulting max fill-ratio across classes if assigned here (lower = more balanced)
            fn = (cur[sp][0] + nn) / target[sp][0] if target[sp][0] > 0 else 1e9
            fa = (cur[sp][1] + na) / target[sp][1] if target[sp][1] > 0 else 1e9
            score = max(fn, fa)
            if best_score is None or score < best_score:
                best, best_score = sp, score
        out[pid] = best
        cur[best][0] += nn; cur[best][1] += na
    return out


class EMBEDDataset(Mammo2DDataset):
    name = 'embed'

    def _build_records(self) -> List[dict]:
        md = pd.read_csv(os.path.join(_TABLES, 'EMBED_OpenData_metadata_reduced.csv'), low_memory=False)
        cl = pd.read_csv(os.path.join(_TABLES, 'EMBED_OpenData_clinical_reduced.csv'), low_memory=False)
        md = md[md['FinalImageType'] == '2D'].copy()
        md['nroi'] = pd.to_numeric(md['num_ROI'], errors='coerce').fillna(0)
        md['side'] = md['ImageLateralityFinal']
        cl['isN'] = (cl['asses'] == 'N')
        j = md.merge(cl[['acc_anon', 'side', 'isN']], on=['acc_anon', 'side'], how='left')
        j = j.dropna(subset=['anon_dicom_path']).drop_duplicates('anon_dicom_path')

        # label: abnormal = ROI>0 ; normal = negative assessment & no ROI ; else drop
        def lab(r):
            if r['nroi'] > 0:
                return 1
            if r['nroi'] == 0 and r['isN'] is True:
                return 0
            return None
        j['label'] = j.apply(lab, axis=1)
        j = j.dropna(subset=['label'])

        # local path (only files actually downloaded)
        j['path'] = _IMG_ROOT + '/' + j['anon_dicom_path'].str.replace(r'^images/', '', regex=True)
        j = j[j['path'].map(os.path.exists)]
        if j.empty:
            return []

        # patient-grouped, image-balanced split (per-patient normal/abnormal image counts)
        pat_counts = defaultdict(lambda: [0, 0])
        for _, r in j.iterrows():
            pat_counts[r['empi_anon']][int(r['label'])] += 1
        split_of = _balanced_group_split(pat_counts)

        recs = []
        for _, r in j.iterrows():
            if split_of[r['empi_anon']] != self.split:
                continue
            recs.append({'image_path': r['path'], 'label': int(r['label']),
                         'uid': os.path.splitext(os.path.basename(r['path']))[0]})
        return recs
