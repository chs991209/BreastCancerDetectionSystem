"""
[Smoke test] 2.5D + 2D-transfer 파이프라인 무결성 검증.
- import 해석, DBT 2.5D [k,Z,H,W] 셰이프, 2D [k,1,H,W] 셰이프,
- 공용 collate → 공용 model(in_chans=k) forward 통과.
모델은 네트워크 없이 pretrained=False 로 생성(셰이프 호환만 검증).
실행: PYTHONPATH=/workspace python smoke_test_25d.py
"""
import torch

from common.model import DimensionAgnostic3DSwin
from fine_tuning.data_preprocessing import BCSDBTDataset, collate_dbt_volumes
from transfer_learning.mammo2d_base import collate_mammo2d
from transfer_learning.vindr.dataset import VinDrMammoDataset
from transfer_learning.cmmd.dataset import CMMDDataset
from transfer_learning.inbreast.dataset import INbreastDataset

R = 1
K = 2 * R + 1
CACHE = '/data/cache'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"=== building shared model (in_chans={K}, pretrained=False) ===")
model = DimensionAgnostic3DSwin(pretrained=False, freeze_backbone=False, in_chans=K).to(device).eval()


def forward_check(tag, batch):
    volumes = batch['volumes'].to(device)
    depth_mask = batch['depth_mask'].to(device)
    with torch.no_grad():
        logits, sal = model(volumes, depth_mask)
    print(f"[{tag}] volumes={tuple(volumes.shape)} depth_mask={tuple(depth_mask.shape)} "
          f"-> logits={tuple(logits.shape)} sal={tuple(sal.shape)}  OK")


# ---- 1) DBT 2.5D ----
print("\n=== DBT 2.5D (val) ===")
dbt = BCSDBTDataset('/data/datasets/BCS-DBT', split='val', cache_dir=CACHE,
                    augment=False, neighbor_slices=R)
v0, m0, y0 = dbt[0]
print(f"item0: vol={tuple(v0.shape)} (expect [{K},Z,384,384]) mask={tuple(m0.shape)} label={y0}")
assert v0.shape[0] == K and v0.ndim == 4, "DBT 2.5D channel/rank mismatch"
forward_check('DBT', collate_dbt_volumes([dbt[0], dbt[1]]))

# ---- 2) 2D transfer datasets ----
for cls in (VinDrMammoDataset, CMMDDataset, INbreastDataset):
    name = cls.name
    print(f"\n=== {name} (test split) ===")
    try:
        ds = cls(split='test', cache_dir=CACHE, neighbor_slices=R, augment=False)
        v, m, y = ds[0]
        print(f"item0: vol={tuple(v.shape)} (expect [{K},1,384,384]) mask={tuple(m.shape)} label={y}")
        assert v.shape == (K, 1, 384, 384), f"{name} 2D shape mismatch: {tuple(v.shape)}"
        forward_check(name, collate_mammo2d([ds[0], ds[1]]))
    except Exception as e:
        print(f"[{name}] FAILED: {type(e).__name__}: {e}")

print("\n=== smoke test complete ===")
