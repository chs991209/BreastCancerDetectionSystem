"""
[DBT-2026 (Segmed) 3D 데이터셋 로더]
former BCS-DBT 와 '동일 출력 규격'((vol[k,Z,H,W], mask[1,Z,H,W], label))으로 방출하여
combined 파이프라인에서 ConcatDataset 로 통합 가능하게 한다. 기존 파일은 건드리지 않음.

디스크 구조 (dataset_root = .../DBT-2026/DBT-2026):
    ecrf/DBT_dataset_eCRF_558_patients_updated.xlsx   # 라벨 (study_id, PathologyType, ...)
    dicoms/<batch>/<StudyUID>/<SeriesUID>/*.dcm        # 뷰별 멀티프레임 볼륨
    annotations/<Segmed_Patient_*>/<StudyUID>.json     # 병변 polygon (v1 미사용)

각 study = 4개 3D 볼륨 뷰: SeriesDescription ∈ {ROUTINE3D_VOL_LCC/RCC/LMLO/RMLO}
(단일프레임 V-Preview/Enhanced 는 제외). 라벨은 환자단위 PathologyType(Malignant=1)을
각 뷰 볼륨에 부여(v1: 대측 유방 라벨 노이즈 감수; polygon 기반 뷰별 정제는 후속).
split: 공식 없음 → crc32(study_id) 결정적 분할(BCS-DBT 와 독립).
캐시: /data/cache/dbt2026/preproc/<split>/<StudyUID>_<view>.npz  (기존 캐시와 분리).
"""
import os
import glob
import json
import zlib
import tempfile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Optional

from common.preprocessing import MedicalImagePreprocessor

VOL_PREFIX = 'ROUTINE3D_VOL_'
VIEWS = ('LCC', 'RCC', 'LMLO', 'RMLO')


def _bucket(key: str, test_frac=0.15, val_frac=0.15) -> str:
    """[구식] crc32 단순 해시 split — 비층화(class 비율이 split마다 흔들림). 사용 안 함."""
    h = (zlib.crc32(key.encode()) % 10000) / 10000.0
    if h < test_frac:
        return 'test'
    if h < test_frac + val_frac:
        return 'val'
    return 'train'


def _stratified_assignment(study_labels: dict, test_frac=0.15, val_frac=0.15) -> dict:
    """[층화(stratified) + 그룹(study 단위) 결정적 split]
    class(abnormal/normal)별로 study 를 crc32 순서로 정렬 후 각 class 안에서 70/15/15 배분 →
    모든 split 이 전체와 동일한 normal:abnormal 비율(층화). study 단위라 뷰 누수 없음(그룹).
    무작위성 없이(정렬+해시) 재현 가능. study_labels: {study_id: abnormal(0/1)}."""
    from collections import defaultdict
    by_class = defaultdict(list)
    for sid, lab in study_labels.items():
        by_class[int(lab)].append(sid)
    assign = {}
    for lab, sids in by_class.items():
        ordered = sorted(sids, key=lambda s: zlib.crc32(s.encode()))
        n = len(ordered)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        for i, s in enumerate(ordered):
            assign[s] = 'test' if i < n_test else ('val' if i < n_test + n_val else 'train')
    return assign


class DBT2026Dataset(Dataset):
    def __init__(self, dataset_root: str, split: str = 'train',
                 target_size: Tuple[int, int] = (384, 384),
                 cache_dir: str = '/data/cache', rebuild_index: bool = False,
                 use_cache: bool = True, augment: bool = False,
                 neighbor_slices: int = 1):
        if split not in ('train', 'val', 'test'):
            raise ValueError(f"split must be train|val|test, got {split!r}")
        self.dataset_root = dataset_root
        self.split = split
        self.augment = augment
        self.use_cache = use_cache
        self.neighbor_slices = neighbor_slices
        self.cache_dir = cache_dir
        self.preproc_dir = os.path.join(cache_dir, 'dbt2026', 'preproc', split)
        self.preprocessor = MedicalImagePreprocessor(target_size=target_size)

        self.ecrf_path = glob.glob(os.path.join(dataset_root, 'ecrf', '*.xlsx'))[0]
        self.dicoms_root = os.path.join(dataset_root, 'dicoms')
        self.index = self._load_or_build_index(rebuild_index)
        n_abn = sum(e['abnormal'] for e in self.index)
        print(f"[DBT2026:{split}] {len(self.index)} view-volumes "
              f"(abnormal/yes-lesion={n_abn}, normal={len(self.index) - n_abn})")

    # ---- index ----
    def _index_cache_path(self) -> str:
        return os.path.join(self.cache_dir, 'dbt2026', f'dbt2026_{self.split}_index.json')

    def _load_or_build_index(self, rebuild: bool) -> List[dict]:
        cf = self._index_cache_path()
        if os.path.exists(cf) and not rebuild:
            with open(cf) as f:
                return json.load(f)
        os.makedirs(os.path.dirname(cf), exist_ok=True)

        # 1) eCRF: study_id → abnormal(yes-lesion) 라벨.
        # 라벨 정의: abnormal = benign-lesion + malignant. 단, PathologyType 은 전원
        # Benign/Malignant 라서 그대로 쓰면 전부 abnormal 이 된다. Group B('normal/benign
        # screeners')는 병변 없는 정상군 → normal. 따라서 abnormal = Group ∈ {A,C,D}.
        # Group B('normal/benign screeners')를 병변 유무로 '적절히' 분할:
        #   B + 병변 polygon 有 → abnormal(benign lesion) / B + polygon 無 → normal(purely no-lesion).
        #   A/C/D 는 생검·콜백 확정 병변이므로 polygon 유무와 무관하게 abnormal.
        # → normal = '병변 어노테이션이 전혀 없는 순수 정상'만 남는다(라벨 노이즈 제거).
        import ast
        ann_root = os.path.join(self.dataset_root, 'annotations')

        def _has_polygon(study_uid: str) -> bool:
            js = glob.glob(os.path.join(ann_root, '*', study_uid + '.json'))
            if not js:
                return False
            try:
                d = json.load(open(js[0]))
            except Exception:
                return False
            a = d.get('annotations', [])
            if isinstance(a, str):
                try:
                    a = ast.literal_eval(a)
                except Exception:
                    return a.strip() not in ('', '[]')
            return isinstance(a, list) and len(a) > 0

        df = pd.read_excel(self.ecrf_path)
        label_of = {}
        for r in df.itertuples(index=False):
            sid = str(r.study_id).strip()
            grp = str(r.Group).strip()
            if grp.startswith('B'):
                label_of[sid] = int(_has_polygon(sid))   # B: 병변 polygon 有 → abnormal
            else:
                label_of[sid] = 1                          # A/C/D → abnormal

        # 1b) 층화+그룹 결정적 split 배정(전체 study 기준 1회 계산 → self.split 만 채택).
        #     class별 70/15/15 → 모든 split 동일 normal:abnormal 비율. study 단위 → 뷰 무누수.
        split_of = _stratified_assignment(label_of, test_frac=0.15, val_frac=0.15)

        # 2) dicoms 트리 스캔: <batch>/<StudyUID>/<SeriesUID>/*.dcm → 3D 볼륨 뷰만 채택
        import pydicom
        index: List[dict] = []
        study_dirs = glob.glob(os.path.join(self.dicoms_root, '*', '*'))
        for sdir in study_dirs:
            study_uid = os.path.basename(sdir)
            if study_uid not in label_of:
                continue  # eCRF 라벨 없는 study 스킵
            if split_of.get(study_uid) != self.split:
                continue
            for series_dir in glob.glob(os.path.join(sdir, '*')):
                dcms = glob.glob(os.path.join(series_dir, '*.dcm'))
                if not dcms:
                    continue
                try:
                    hdr = pydicom.dcmread(dcms[0], stop_before_pixels=True)
                except Exception:
                    continue
                desc = str(getattr(hdr, 'SeriesDescription', ''))
                nframes = int(getattr(hdr, 'NumberOfFrames', 1) or 1)
                if not desc.startswith(VOL_PREFIX) or nframes <= 1:
                    continue  # 프리뷰/단일프레임 제외
                view = desc[len(VOL_PREFIX):].strip().upper()
                index.append({
                    'study_uid': study_uid, 'view': view,
                    'abnormal': label_of[study_uid], 'dcm_path': dcms[0],
                })
        with open(cf, 'w') as f:
            json.dump(index, f)
        return index

    def __len__(self):
        return len(self.index)

    # ---- preprocessing cache (reuse BCS-DBT pattern) ----
    def _preproc_cache_path(self, e: dict) -> str:
        return os.path.join(self.preproc_dir, f"{e['study_uid']}_{e['view']}.npz")

    def _compute(self, e: dict) -> np.ndarray:
        vol = self.preprocessor.read_volume(e['dcm_path'])       # [Z,H,W] uint8
        crop = self.preprocessor.compute_crop_rect(vol)
        return np.stack([self.preprocessor.finalize_image(vol[z], crop)
                         for z in range(vol.shape[0])], axis=0)   # [Z,384,384] uint8

    def _load_or_build(self, e: dict) -> np.ndarray:
        if not self.use_cache:
            return self._compute(e)
        cp = self._preproc_cache_path(e)
        if os.path.exists(cp):
            try:
                with np.load(cp) as d:
                    return d['img']
            except Exception:
                pass
        img = self._compute(e)
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        tmp = f"{cp}.tmp.{os.getpid()}"
        try:
            with open(tmp, 'wb') as fh:
                np.savez_compressed(fh, img=img)
            os.replace(tmp, cp)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
        return img

    def _make_2p5d(self, img_vol: np.ndarray) -> np.ndarray:
        """[2.5D] [Z,H,W] → [k,Z,H,W] (k=2r+1), edge-clamped Z-shifts. BCS-DBT 와 동일."""
        r = self.neighbor_slices
        if r <= 0:
            return img_vol[np.newaxis, ...]
        z = img_vol.shape[0]
        base = np.arange(z)
        return np.stack([img_vol[np.clip(base + d, 0, z - 1)] for d in range(-r, r + 1)], axis=0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        e = self.index[idx]
        img_vol = self._load_or_build(e)                              # [Z,384,384] uint8
        vol = torch.from_numpy(self._make_2p5d(img_vol)).float() / 255.0   # [k,Z,H,W]
        # DBT-2026 v1: 슬라이스 박스 미사용 → 영마스크(Dice 비활성). [1,Z,H,W].
        mask = torch.zeros((1, vol.shape[1], vol.shape[2], vol.shape[3]), dtype=torch.float32)
        if self.augment:
            if torch.rand(1).item() < 0.5:
                vol = torch.flip(vol, dims=[-1]); mask = torch.flip(mask, dims=[-1])
            c = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
            b = (torch.rand(1).item() - 0.5) * 0.1
            vol = torch.clamp((vol - 0.5) * c + 0.5 + b, 0.0, 1.0)
        return vol, mask, e['abnormal']
