"""
[2D Transfer-Learning 공용 베이스]
VinDr-Mammo / CMMD / INbreast 2D FFDM 데이터셋의 공통 로직(단일프레임 DICOM 디코딩,
SSD 캐시, 2.5D 포맷 방출, 결정적 split)을 담는다. 서브클래스(각 dataset.py)는
`_build_records()` 하나만 구현한다.

핵심 설계: 2D 이미지를 3D DBT 파이프라인과 '동일 포맷'으로 방출한다.
  - 단일 2D 이미지를 k(=2r+1)채널로 복제 → 퇴화된 2.5D → [k, 1, H, W] (Z=1 볼륨).
  - 마스크는 [1, 1, H, W] 영행렬(2D transfer 는 박스 미사용, 분류 손실만).
  - collate_mammo2d 가 model(volumes, depth_mask) 규격의 dict 를 만든다.
따라서 common.model.DimensionAgnostic3DSwin(in_chans=k) 을 3D/2D 양쪽에 그대로 재사용.
"""
import os
import json
import zlib
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple

from common.preprocessing import MedicalImagePreprocessor


def collate_mammo2d(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]) -> Dict[str, torch.Tensor]:
    """2D 배치 → model(volumes, depth_mask) 규격. 전부 동일 [k,1,H,W] 라 단순 stack.
    반환: volumes[B,k,1,H,W], masks[B,1,1,H,W], depth_mask[B,1]=True, labels[B]."""
    vols, masks, labels = zip(*batch)
    volumes = torch.stack(vols, 0)                          # [B, k, 1, H, W]
    masks = torch.stack(masks, 0)                           # [B, 1, 1, H, W]
    depth_mask = torch.ones((len(batch), 1), dtype=torch.bool)  # Z=1, 전부 유효
    return {
        'volumes': volumes,
        'masks': masks,
        'depth_mask': depth_mask,
        'labels': torch.tensor(labels, dtype=torch.long),
    }


def deterministic_bucket(key: str, test_frac: float = 0.2, val_frac: float = 0.1) -> str:
    """crc32 기반 결정적 split(재현성·무누수). key(환자ID 권장)로 train/val/test 배정.
    공식 split 이 없는 CMMD/INbreast 및 VinDr val 홀드아웃에 사용."""
    h = (zlib.crc32(key.encode()) % 10000) / 10000.0
    if h < test_frac:
        return 'test'
    if h < test_frac + val_frac:
        return 'val'
    return 'train'


class Mammo2DDataset(Dataset):
    """2D FFDM transfer-learning 베이스. 서브클래스는 name/ _build_records() 지정."""
    name = 'mammo2d'  # 서브클래스에서 override ('vindr'|'cmmd'|'inbreast')

    def __init__(self, split: str = 'train',
                 cache_dir: str = '/data/cache',
                 target_size: Tuple[int, int] = (384, 384),
                 neighbor_slices: int = 1, augment: bool = False,
                 use_cache: bool = True, rebuild_index: bool = False):
        if split not in ('train', 'val', 'test'):
            raise ValueError(f"split must be train|val|test, got {split!r}")
        self.split = split
        # k=2r+1: 3D DBT 와 동일한 in_chans 를 맞춰야 같은 모델에 진입 가능.
        self.neighbor_slices = neighbor_slices
        self.augment = augment
        self.use_cache = use_cache
        self.target_size = target_size
        self.cache_dir = cache_dir
        self.preproc_dir = os.path.join(cache_dir, self.name, 'preproc', split)
        self.preprocessor = MedicalImagePreprocessor(target_size=target_size)

        self.records = self._load_or_build_index(rebuild_index)
        n_pos = sum(int(r['label']) for r in self.records)
        print(f"[{self.name}:{split}] {len(self.records)} images (pos={n_pos}, "
              f"neg={len(self.records) - n_pos})")

    # ---- 서브클래스 구현부 ----
    def _build_records(self) -> List[dict]:
        """[{'image_path': str, 'label': 0|1, 'uid': str}] 리스트 반환 (해당 split 만)."""
        raise NotImplementedError

    # ---- 인덱스 캐시 ----
    def _index_cache_path(self) -> str:
        return os.path.join(self.cache_dir, self.name, f'{self.name}_{self.split}_index.json')

    def _load_or_build_index(self, rebuild: bool) -> List[dict]:
        cf = self._index_cache_path()
        if os.path.exists(cf) and not rebuild:
            with open(cf) as f:
                return json.load(f)
        os.makedirs(os.path.dirname(cf), exist_ok=True)
        recs = self._build_records()
        if not recs:
            raise RuntimeError(f"[{self.name}:{self.split}] no records built — check paths/labels")
        with open(cf, 'w') as f:
            json.dump(recs, f)
        return recs

    def __len__(self) -> int:
        return len(self.records)

    # ---- 전처리 캐시(이미지 단위 npz) ----
    def _preproc_cache_path(self, rec: dict) -> str:
        return os.path.join(self.preproc_dir, f"{rec['uid']}.npz")

    def _compute_preproc(self, rec: dict) -> np.ndarray:
        vol = self.preprocessor.read_volume(rec['image_path'])  # 단일프레임 → [1, H, W] uint8
        crop = self.preprocessor.compute_crop_rect(vol)
        return self.preprocessor.finalize_image(vol[0], crop)   # [th, tw] uint8

    def _load_or_build(self, rec: dict) -> np.ndarray:
        if not self.use_cache:
            return self._compute_preproc(rec)
        cp = self._preproc_cache_path(rec)
        if os.path.exists(cp):
            try:
                with np.load(cp) as d:
                    return d['img']
            except Exception:
                pass  # 손상 캐시 → 재생성
        img = self._compute_preproc(rec)
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

    def _augment(self, vol: torch.Tensor, mask: torch.Tensor):
        """[train 증강] vol[k,1,H,W]/mask[1,1,H,W] float[0,1]. hflip(W축) + 밝기/대비 지터."""
        if torch.rand(1).item() < 0.5:
            vol = torch.flip(vol, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
        contrast = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
        bright = (torch.rand(1).item() - 0.5) * 0.1
        vol = torch.clamp((vol - 0.5) * contrast + 0.5 + bright, 0.0, 1.0)
        return vol, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        rec = self.records[idx]
        img = self._load_or_build(rec)                 # uint8 [H, W]
        k = 2 * self.neighbor_slices + 1
        t = torch.from_numpy(img).float() / 255.0      # [H, W]
        # 2D → 퇴화된 2.5D: k채널 복제 + Z=1 → [k, 1, H, W] (3D 볼륨과 동일 랭크)
        vol = t.unsqueeze(0).unsqueeze(0).repeat(k, 1, 1, 1)          # [k, 1, H, W]
        mask = torch.zeros((1, 1, t.shape[-2], t.shape[-1]), dtype=torch.float32)
        if self.augment:
            vol, mask = self._augment(vol, mask)
        return vol, mask, int(rec['label'])
