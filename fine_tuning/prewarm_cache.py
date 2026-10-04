"""
BCS-DBT 전처리 캐시 프리워밍(pre-warm) 유틸리티.

모든 볼륨을 1회 디코딩/전처리하여 외장 SSD(/data/cache)에 .npz 로 저장한다.
- 멀티 워커 병렬 디코딩 (JPEG2000 디코딩이 CPU 바운드)
- 재실행 안전: 이미 캐시된 볼륨은 즉시 스킵되므로 중단 후 이어서 가능
- 워커는 인덱스(int)만 반환 → 대용량 텐서 IPC 회피

사용:
    python prewarm_cache.py                  # val -> train -> test 순서
    python prewarm_cache.py train            # 특정 split 만
    python prewarm_cache.py train --workers 32
"""
import sys
import time
import argparse
from torch.utils.data import Dataset, DataLoader
from fine_tuning.data_preprocessing import BCSDBTDataset

DATASET_ROOT = "/data/datasets/BCS-DBT"


def _take_first(batch):
    """batch_size=1 의 단일 인덱스만 반환 (대용량 텐서 IPC 회피)."""
    return batch[0]


class _PrewarmView(Dataset):
    """__getitem__ 가 디코딩+캐시 저장만 수행하고 인덱스만 돌려주는 얇은 래퍼."""
    def __init__(self, ds: BCSDBTDataset):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        self.ds._load_or_build(self.ds.index[i])  # side-effect: SSD 캐시 채우기
        return i


def prewarm(split: str, num_workers: int):
    print(f"\n=== prewarm split={split} (workers={num_workers}) ===", flush=True)
    ds = BCSDBTDataset(dataset_root=DATASET_ROOT, split=split)
    n = len(ds)
    loader = DataLoader(_PrewarmView(ds), batch_size=1, shuffle=False,
                        num_workers=num_workers, collate_fn=_take_first)

    t0 = time.time()
    for done, _ in enumerate(loader, 1):
        if done % 50 == 0 or done == n:
            el = time.time() - t0
            rate = el / done
            eta = (n - done) * rate / 60.0
            print(f"[{split}] {done}/{n}  {rate:.2f}s/vol  elapsed {el/60:.1f}min  ETA {eta:.0f}min",
                  flush=True)
    print(f"[{split}] DONE {n} volumes in {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("splits", nargs="*", default=["val", "train", "test"],
                    help="splits to prewarm (default: val train test)")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    # 인자가 있으면 그것만, 없으면 기본 순서(빠른 val 먼저)
    splits = args.splits if args.splits else ["val", "train", "test"]
    print(f"[prewarm] splits={splits} workers={args.workers}", flush=True)
    for s in splits:
        prewarm(s, args.workers)
    print("[prewarm] ALL SPLITS COMPLETE", flush=True)
