import os
import glob
import json
import tempfile
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Sampler
from typing import Union, Tuple, List, Dict, Optional
from tqdm import tqdm

# [공용 전처리] MedicalImagePreprocessor 는 common 으로 분리(2D transfer_learning 과 공유).
from common.preprocessing import MedicalImagePreprocessor


class BCSDBTDataset(Dataset):
    """
    Duke BCS-DBT 공식 메타데이터(라벨/박스/파일경로 CSV) 기반 3D 볼륨 데이터셋.

    실제 디스크 구조 (dataset_root = .../BCS-DBT):
        Metadata/   BCS-DBT-{labels,boxes,file-paths}-{train,validation,test}*.csv
        {Train,Val,Test}/breast_cancer_screening_dbt/<PID>/<StudyUID>/<SeriesUID>/<uuid>.dcm
    각 (PID, StudyUID, View) = 1개 멀티프레임 DICOM = 1개 3D 볼륨([Z,H,W]).

    file-paths CSV 의 classic_path 로 (PID/StudyUID/SeriesUID) 를 얻어 시리즈 폴더를
    찾고, 그 안의 단일 .dcm 을 glob 으로 해석한다(디스크 파일명은 UUID 이므로 glob 필요).
    해석 결과 인덱스는 cache_dir 에 JSON 으로 캐싱(느린 외장 SSD glob 1회만 수행).
    """

    # split -> (디스크 하위폴더, 라벨 CSV, 박스 CSV, 파일경로 CSV)
    SPLIT_CFG = {
        'train': ('Train', 'BCS-DBT-labels-train-v2.csv',
                  'BCS-DBT-boxes-train-v2.csv', 'BCS-DBT-file-paths-train-v2.csv'),
        'val':   ('Val', 'BCS-DBT-labels-validation-PHASE-2-Jan-2024.csv',
                  'BCS-DBT-boxes-validation-v2-PHASE-2-Jan-2024.csv', 'BCS-DBT-file-paths-validation-v2.csv'),
        'test':  ('Test', 'BCS-DBT-labels-test-PHASE-2.csv',
                  'BCS-DBT-boxes-test-v2-PHASE-2-Jan-2024.csv', 'BCS-DBT-file-paths-test-v2.csv'),
    }

    def __init__(self, dataset_root: str, split: str = 'train',
                 target_size: Tuple[int, int] = (384, 384),
                 cache_dir: Optional[str] = None, rebuild_index: bool = False,
                 use_cache: bool = True, augment: bool = False,
                 neighbor_slices: int = 1):
        split = {'validation': 'val'}.get(split, split)
        if split not in self.SPLIT_CFG:
            raise ValueError(f"split must be one of {list(self.SPLIT_CFG)}, got {split!r}")

        self.dataset_root = dataset_root
        self.split = split
        split_dir, labels_csv, boxes_csv, paths_csv = self.SPLIT_CFG[split]

        self.meta_dir = os.path.join(dataset_root, 'Metadata')
        self.image_root = os.path.join(dataset_root, split_dir, 'breast_cancer_screening_dbt')
        self.labels_path = os.path.join(self.meta_dir, labels_csv)
        self.boxes_path = os.path.join(self.meta_dir, boxes_csv)
        self.paths_path = os.path.join(self.meta_dir, paths_csv)
        self.cache_dir = self._resolve_cache_dir(cache_dir)

        # 전처리된 볼륨 캐시(외장 SSD). 첫 접근 시 DICOM 디코딩(~15s)을 1회만 수행하고
        # uint8 [Z, H, W] 로 압축 저장 → 이후 에폭은 즉시 로드. OS 디스크가 아닌 SSD 에 적재.
        self.use_cache = use_cache
        self.preproc_dir = os.path.join(self.cache_dir, 'preproc', split)

        # 온더플라이 증강(train 전용): 균형 배치 샘플러가 76개 양성을 반복 노출하므로
        # 매 접근마다 다른 변형을 가해 '동일 양성 반복'에 의한 과적합을 완화한다.
        self.augment = augment

        # [2.5D 인접 슬라이스 채널] 각 슬라이스 위치 z 를 [z-r .. z+r] (k=2r+1) 채널 스택으로
        # 구성해 층간(through-plane) 문맥을 '입력'에 주입한다. r=0 이면 기존 1채널과 동일.
        # 2D transfer_learning(VinDr/CMMD/INbreast)는 k=1 복제로 동일 Swin(in_chans=k)에 진입.
        self.neighbor_slices = neighbor_slices

        self.preprocessor = MedicalImagePreprocessor(target_size=target_size)

        # 박스 룩업: (PID, StudyUID, View) -> {slice_idx: [(x,y,w,h), ...]}
        self.boxes_lookup = self._build_boxes_lookup()
        # 볼륨 인덱스: 디스크에 실제로 존재하는 (라벨 포함) 볼륨만
        self.index = self._build_index(rebuild_index)
        print(f"[BCS-DBT:{split}] indexed {len(self.index)} volumes "
              f"(cancer+={sum(e['cancer'] for e in self.index)})")

    @staticmethod
    def _resolve_cache_dir(cache_dir: Optional[str]) -> str:
        """쓰기 가능한 캐시 디렉토리 확보 (원본 데이터셋은 read-only 마운트일 수 있음)."""
        candidates = [cache_dir, os.environ.get('BCSDBT_CACHE'),
                      '/data/cache', os.path.join(tempfile.gettempdir(), 'bcsdbt_cache')]
        for c in candidates:
            if not c:
                continue
            try:
                os.makedirs(c, exist_ok=True)
                test = os.path.join(c, '.write_test')
                with open(test, 'w') as f:
                    f.write('ok')
                os.remove(test)
                return c
            except OSError:
                continue
        raise RuntimeError("No writable cache_dir found for BCS-DBT index")

    def _build_boxes_lookup(self) -> Dict[Tuple[str, str, str], Dict[int, List[Tuple[int, int, int, int]]]]:
        boxes_df = pd.read_csv(self.boxes_path)
        lookup: Dict[Tuple[str, str, str], Dict[int, list]] = {}
        for row in boxes_df.itertuples(index=False):
            key = (row.PatientID, row.StudyUID, row.View)
            slc = int(row.Slice)
            lookup.setdefault(key, {}).setdefault(slc, []).append(
                (int(row.X), int(row.Y), int(row.Width), int(row.Height)))
        return lookup

    def _build_index(self, rebuild: bool) -> List[dict]:
        # v2: 4-클래스(Normal/Actionable/Benign/Cancer) 정보 포함 (언더샘플링 샘플러용)
        cache_file = os.path.join(self.cache_dir, f'bcsdbt_{self.split}_index_v2.json')
        if os.path.exists(cache_file) and not rebuild:
            with open(cache_file) as f:
                return json.load(f)

        print(f"[BCS-DBT:{self.split}] building volume index (one-time glob over SSD)...")
        paths_df = pd.read_csv(self.paths_path)
        label_cols = ['PatientID', 'StudyUID', 'View', 'Normal', 'Actionable', 'Benign', 'Cancer']
        labels_df = pd.read_csv(self.labels_path)[label_cols]
        merged = paths_df.merge(labels_df, on=['PatientID', 'StudyUID', 'View'], how='inner')

        index: List[dict] = []
        missing = 0
        # disable=None → 비-TTY(로그 리다이렉트)에서는 진행바 자동 비활성 (로그 폭주/EFAULT 방지)
        for row in tqdm(merged.itertuples(index=False), total=len(merged), disable=None):
            # classic_path: Breast-Cancer-Screening-DBT/<PID>/<StudyUID>/<SeriesUID>/1-1.dcm
            parts = row.classic_path.split('/')
            pid, study_uid, series_uid = parts[1], parts[2], parts[3]
            series_dir = os.path.join(self.image_root, pid, study_uid, series_uid)
            dcms = glob.glob(os.path.join(series_dir, '*.dcm'))
            if not dcms:            # 다운로드 누락된 볼륨(예: 빈 환자 폴더)은 스킵
                missing += 1
                continue
            # 4-클래스 라벨은 상호 배타적(one-hot). 우선순위로 단일 klass 결정.
            if int(row.Cancer):
                klass = 'cancer'
            elif int(row.Benign):
                klass = 'benign'
            elif int(row.Actionable):
                klass = 'actionable'
            else:
                klass = 'normal'
            index.append({
                'patient_id': row.PatientID, 'study_uid': row.StudyUID, 'view': row.View,
                'cancer': int(row.Cancer), 'klass': klass, 'dcm_path': dcms[0],
            })
        print(f"[BCS-DBT:{self.split}] resolved {len(index)} volumes, {missing} missing on disk")

        with open(cache_file, 'w') as f:
            json.dump(index, f)
        return index

    def __len__(self):
        return len(self.index)

    def _preproc_cache_path(self, entry: dict) -> str:
        # 환자별 하위폴더로 분산 저장(exfat 단일 폴더 과밀 방지)
        return os.path.join(self.preproc_dir, entry['patient_id'],
                            f"{entry['study_uid']}_{entry['view']}.npz")

    def _compute_preproc_volume(self, entry: dict) -> Tuple[np.ndarray, np.ndarray]:
        """DICOM 디코딩 + 전처리 → (img_vol, mask_vol) 둘 다 uint8 [Z, th, tw]."""
        volume = self.preprocessor.read_volume(entry['dcm_path'])  # [Z, H, W] uint8
        z_dim, h, w = volume.shape
        crop_rect = self.preprocessor.compute_crop_rect(volume)
        slice_to_boxes = self.boxes_lookup.get(
            (entry['patient_id'], entry['study_uid'], entry['view']), {})

        img_slices, mask_slices = [], []
        for z in range(z_dim):
            mask_full = np.zeros((h, w), dtype=np.uint8)
            for (bx, by, bw, bh) in slice_to_boxes.get(z, []):
                cv2.rectangle(mask_full, (bx, by), (bx + bw, by + bh), 255, -1)
            img_slices.append(self.preprocessor.finalize_image(volume[z], crop_rect))
            mask_slices.append(self.preprocessor.finalize_mask(mask_full, crop_rect))
        return np.stack(img_slices, axis=0), np.stack(mask_slices, axis=0)  # [Z, th, tw]

    def _load_or_build(self, entry: dict) -> Tuple[np.ndarray, np.ndarray]:
        """캐시가 있으면 즉시 로드, 없으면 디코딩 후 SSD 에 원자적 저장."""
        if not self.use_cache:
            return self._compute_preproc_volume(entry)

        cache_path = self._preproc_cache_path(entry)
        if os.path.exists(cache_path):
            try:
                with np.load(cache_path) as data:
                    return data['img'], data['mask']
            except Exception:
                pass  # 손상 캐시 → 재생성

        img_vol, mask_vol = self._compute_preproc_volume(entry)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        # 워커 경합/부분쓰기 방지: 임시파일에 쓰고 원자적 rename
        tmp = f"{cache_path}.tmp.{os.getpid()}"
        try:
            # 파일 핸들로 저장(문자열 경로는 numpy 가 .npz 를 덧붙여 rename 이 깨짐)
            with open(tmp, 'wb') as fh:
                np.savez_compressed(fh, img=img_vol, mask=mask_vol)
            os.replace(tmp, cache_path)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
        return img_vol, mask_vol

    def _augment_volume(self, vol: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """[train 온더플라이 증강] vol: [k, Z, H, W], mask: [1, Z, H, W] float[0,1].
        좌우 반전은 W축(-1)이라 채널 수 k 와 무관하게 안전. 워커마다 RNG 분리됨."""
        # 좌우 반전(유방촬영에서 기하학적으로 유효). 이미지·마스크 동시 적용.
        if torch.rand(1).item() < 0.5:
            vol = torch.flip(vol, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
        # 밝기/대비 지터 (이미지에만; 마스크는 라벨이므로 제외)
        contrast = 1.0 + (torch.rand(1).item() - 0.5) * 0.2   # [0.9, 1.1]
        bright = (torch.rand(1).item() - 0.5) * 0.1            # [-0.05, 0.05]
        vol = torch.clamp((vol - 0.5) * contrast + 0.5 + bright, 0.0, 1.0)
        return vol, mask

    def _make_2p5d(self, img_vol: np.ndarray) -> np.ndarray:
        """[2.5D 스택] uint8 [Z, H, W] → [k, Z, H, W] (k=2r+1). 채널 c=d+r 은
        볼륨을 Z축으로 d 만큼 이동(경계는 edge-clamp)한 슬라이스. r=0 → [1, Z, H, W].
        캐시(디스크 [Z,H,W])는 그대로 두고 로드 시점에 이웃을 모아 채널화 → 재디코딩 불필요."""
        r = self.neighbor_slices
        if r <= 0:
            return img_vol[np.newaxis, ...]  # [1, Z, H, W]
        z_dim = img_vol.shape[0]
        base = np.arange(z_dim)
        # 각 오프셋 d 에 대해 clamp(z+d) 인덱스로 슬라이스를 모아 채널축(axis=0)으로 스택
        chans = [img_vol[np.clip(base + d, 0, z_dim - 1)] for d in range(-r, r + 1)]
        return np.stack(chans, axis=0)  # [k, Z, H, W]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        entry = self.index[idx]
        img_vol, mask_vol = self._load_or_build(entry)  # uint8 [Z, th, tw]

        # uint8 [Z, H, W] → 2.5D float 텐서 [k, Z, H, W] (0~1 스케일). 마스크는 중심 1채널 유지.
        volume_tensor = torch.from_numpy(self._make_2p5d(img_vol)).float() / 255.0
        mask_volume = torch.from_numpy(mask_vol).float().unsqueeze(0) / 255.0

        if self.augment:
            volume_tensor, mask_volume = self._augment_volume(volume_tensor, mask_volume)
        return volume_tensor, mask_volume, entry['cancer']


def collate_dbt_volumes(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]) -> Dict[str, torch.Tensor]:
    """
    [3D 배치 콜레이트: 가변 깊이(Z) 정합]
    DBT 볼륨은 시리즈마다 슬라이스 수(Z)가 달라 기본 default_collate 로는 배치가 불가능합니다.
    배치 내 최대 깊이(Z_max)에 맞춰 0-패딩하고, 어떤 슬라이스가 실제 데이터인지 알려주는
    depth_mask 를 함께 반환합니다 (Transformer attention mask / 패딩 슬라이스 손실 제외용).

    입력 item: (volume[k, Z, H, W], mask[1, Z, H, W], label:int)   # k=2.5D 채널(2r+1)
    반환:
        volumes    : [B, k, Z_max, H, W]
        masks      : [B, 1, Z_max, H, W]
        depth_mask : [B, Z_max]  (True = 실제 슬라이스, False = 패딩)
        labels     : [B]
    """
    volumes, masks, labels = zip(*batch)

    b = len(batch)
    c = volumes[0].shape[0]                             # 2.5D 채널 수 k (DBT·2D 공통 in_chans)
    h, w = volumes[0].shape[-2], volumes[0].shape[-1]   # [k, Z, H, W]
    z_max = max(v.shape[1] for v in volumes)

    batched_vol = torch.zeros((b, c, z_max, h, w), dtype=volumes[0].dtype)
    batched_mask = torch.zeros((b, 1, z_max, h, w), dtype=masks[0].dtype)  # 마스크는 항상 1채널
    depth_mask = torch.zeros((b, z_max), dtype=torch.bool)

    for i, (vol, msk) in enumerate(zip(volumes, masks)):
        z = vol.shape[1]
        batched_vol[i, :, :z] = vol
        batched_mask[i, :, :z] = msk
        depth_mask[i, :z] = True

    return {
        'volumes': batched_vol,
        'masks': batched_mask,
        'depth_mask': depth_mask,
        'labels': torch.tensor(labels, dtype=torch.long),
    }


class UndersampleNormalSampler(Sampler):
    """
    [클래스 불균형 해소 — Data Layer: 전략적 언더샘플링]
    BCS-DBT train 은 Normal 이 18232/19148(95%)로 압도적. 매 epoch:
      - 소견/박스가 있는 모든 케이스(Actionable+Benign+Cancer)는 100% 투입
      - Normal 은 무작위로 normal_fraction(기본 12%)만 추출
    하여 배치 내 양성 비율을 인위적으로 끌어올린다. Normal 부분집합은 epoch 마다
    새로 추출되어(전역 RNG) 학습 전반에 더 다양한 음성을 노출한다.

    오버샘플링(복원추출)과 달리 희소 양성을 반복 노출하지 않아 과적합 위험이 낮다.
    (잔여 불균형은 BCEWithLogitsLoss 의 pos_weight 로 보정 — Train Layer 참고)
    """
    def __init__(self, dataset: "BCSDBTDataset", normal_fraction: float = 0.12):
        self.normal_fraction = normal_fraction
        klasses = [e['klass'] for e in dataset.index]
        self.normal_idx = [i for i, k in enumerate(klasses) if k == 'normal']
        self.keep_idx = [i for i, k in enumerate(klasses) if k != 'normal']  # 소견 케이스 100%
        self.n_normal_keep = max(1, int(round(len(self.normal_idx) * normal_fraction)))

        # 손실 가중치 산출용: 양성(cancer) 수와 epoch당 음성 수
        self.n_pos = sum(int(dataset.index[i]['cancer'] == 1) for i in range(len(dataset.index)))
        self.epoch_len = len(self.keep_idx) + self.n_normal_keep
        self.n_neg = self.epoch_len - self.n_pos
        print(f"[Sampler] undersample-Normal: keep(non-normal)={len(self.keep_idx)} "
              f"+ normal {self.n_normal_keep}/{len(self.normal_idx)} ({normal_fraction:.0%}) "
              f"→ epoch={self.epoch_len}, pos={self.n_pos} ({100*self.n_pos/self.epoch_len:.2f}%)")

    @property
    def pos_weight(self) -> float:
        return self.n_neg / max(self.n_pos, 1)

    def __len__(self) -> int:
        return self.epoch_len

    def __iter__(self):
        # epoch 마다 Normal 부분집합 새로 추출 (전역 RNG → epoch 간 변동)
        perm = torch.randperm(len(self.normal_idx))[:self.n_normal_keep]
        chosen_normal = [self.normal_idx[p] for p in perm.tolist()]
        epoch_idx = self.keep_idx + chosen_normal
        order = torch.randperm(len(epoch_idx)).tolist()
        return iter([epoch_idx[o] for o in order])


def make_undersample_sampler(dataset: "BCSDBTDataset",
                             normal_fraction: float = 0.12) -> UndersampleNormalSampler:
    """전략적 언더샘플링 샘플러 팩토리 (Normal 만 normal_fraction 추출, 나머지 100%)."""
    return UndersampleNormalSampler(dataset, normal_fraction=normal_fraction)


class BalancedBatchSampler(Sampler):
    """
    [클래스 불균형 해소 — 강화판: 고정 비율 균형 배치 샘플러]
    매 배치마다 양성:음성 개수를 '강제 고정'한다 (예: batch_size=4, pos_fraction=0.5
    → 배치당 양성 2 + 음성 2). 언더샘플링만으로는 양성율이 2.45%에 그쳐 대부분 배치가
    음성뿐이던 문제를, 모든 배치에 양성이 반드시 포함되도록 보장하여 해소한다.

    - 음성: hard-neg(actionable+benign) 100% + Normal 의 normal_fraction 만 추출 →
      epoch 마다 새로 섞어 1회 순회(다양한 음성 노출, 음성 언더샘플링).
    - 양성: 76개를 셔플·순환(부족분 재사용)하되, BCSDBTDataset(augment=True) 의
      온더플라이 증강으로 매번 다른 변형이 적용되어 반복 과적합을 완화.
    - 배치가 이미 균형이므로 pos_weight 불필요(이중 보정 방지) → 표준 BCE 사용.

    DataLoader(batch_sampler=...) 로 전달 (batch_size/shuffle/sampler 와 상호 배타).
    """
    def __init__(self, dataset: "BCSDBTDataset", batch_size: int = 4,
                 pos_fraction: float = 0.5, normal_fraction: float = 0.12):
        idx = dataset.index
        self.pos_idx = [i for i, e in enumerate(idx) if e['cancer'] == 1]
        self.hard_neg_idx = [i for i, e in enumerate(idx) if e['klass'] in ('actionable', 'benign')]
        self.normal_idx = [i for i, e in enumerate(idx) if e['klass'] == 'normal']
        if not self.pos_idx:
            raise ValueError("BalancedBatchSampler: no positive (cancer) samples in dataset")

        self.batch_size = batch_size
        self.pos_per_batch = max(1, round(batch_size * pos_fraction))
        self.neg_per_batch = max(1, batch_size - self.pos_per_batch)
        self.normal_fraction = normal_fraction
        self.n_normal_keep = max(1, int(round(len(self.normal_idx) * normal_fraction)))

        neg_pool_size = len(self.hard_neg_idx) + self.n_normal_keep
        self.num_batches = max(1, neg_pool_size // self.neg_per_batch)
        print(f"[Sampler] balanced-batch: batch={batch_size} "
              f"(pos {self.pos_per_batch} + neg {self.neg_per_batch}) × {self.num_batches} batches "
              f"| neg pool = hardneg {len(self.hard_neg_idx)} + normal {self.n_normal_keep} "
              f"| {len(self.pos_idx)} unique pos cycled (+augment)")

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        # 음성 풀: hard-neg 전체 + Normal 부분추출 → epoch 마다 새로 섞기
        norm_pick = [self.normal_idx[p]
                     for p in torch.randperm(len(self.normal_idx))[:self.n_normal_keep].tolist()]
        neg_pool = self.hard_neg_idx + norm_pick
        neg_order = [neg_pool[p] for p in torch.randperm(len(neg_pool)).tolist()]

        # 양성 순환 큐 (소진 시 재셔플)
        pos_queue: List[int] = []

        def draw_pos(k: int) -> List[int]:
            out: List[int] = []
            while len(out) < k:
                if not pos_queue:
                    pos_queue.extend(self.pos_idx[p]
                                     for p in torch.randperm(len(self.pos_idx)).tolist())
                out.append(pos_queue.pop())
            return out

        nptr = 0
        for _ in range(self.num_batches):
            negs = neg_order[nptr:nptr + self.neg_per_batch]
            nptr += self.neg_per_batch
            batch = draw_pos(self.pos_per_batch) + negs
            order = torch.randperm(len(batch)).tolist()
            yield [batch[o] for o in order]


def make_balanced_batch_sampler(dataset: "BCSDBTDataset", batch_size: int = 4,
                                pos_fraction: float = 0.5,
                                normal_fraction: float = 0.12) -> BalancedBatchSampler:
    """고정 비율 균형 배치 샘플러 팩토리 (모든 배치에 양성 보장)."""
    return BalancedBatchSampler(dataset, batch_size=batch_size,
                                pos_fraction=pos_fraction, normal_fraction=normal_fraction)


class RandomRatioUndersampler(Sampler):
    """
    [불균형 완화 — 데이터셋 수준 무작위 언더샘플링(목표 양성비)]
    매 epoch: 양성(cancer) 전량 + 전체 음성에서 무작위로 n_neg_keep 개만 추출하여
    학습 풀의 양성 비율을 목표치(pos_rate)로 맞춘다. 음성 '총량'을 줄이는 실험용
    (BalancedBatchSampler 와 달리 배치별 양성 보장 없음 — 순수 무작위 다운샘플링).
    잔여 불균형(≈1:(1-p)/p)은 focal loss 의 alpha 로 양성 up-weight (pos_weight 병용 가능).
    """
    def __init__(self, dataset: "BCSDBTDataset", pos_rate: float = 0.08):
        idx = dataset.index
        self.pos_idx = [i for i, e in enumerate(idx) if e['cancer'] == 1]
        self.neg_idx = [i for i, e in enumerate(idx) if e['cancer'] != 1]
        if not self.pos_idx:
            raise ValueError("RandomRatioUndersampler: no positive (cancer) samples")
        self.pos_rate = pos_rate
        self.n_pos = len(self.pos_idx)
        # 양성이 전체의 pos_rate 가 되도록 유지할 음성 수 (전체 음성보다 클 수 없음)
        n_neg_keep = int(round(self.n_pos * (1.0 - pos_rate) / pos_rate))
        self.n_neg_keep = min(n_neg_keep, len(self.neg_idx))
        self.epoch_len = self.n_pos + self.n_neg_keep
        print(f"[Sampler] random-ratio undersample: pos={self.n_pos} "
              f"+ neg {self.n_neg_keep}/{len(self.neg_idx)} → epoch={self.epoch_len}, "
              f"pos_rate={100 * self.n_pos / self.epoch_len:.1f}% (target {100 * pos_rate:.0f}%)")

    @property
    def pos_weight(self) -> float:
        return self.n_neg_keep / max(self.n_pos, 1)

    def __len__(self) -> int:
        return self.epoch_len

    def __iter__(self):
        # epoch 마다 음성 부분집합 새로 무작위 추출(전역 RNG → 다양한 음성 노출)
        perm = torch.randperm(len(self.neg_idx))[:self.n_neg_keep]
        chosen_neg = [self.neg_idx[p] for p in perm.tolist()]
        epoch_idx = self.pos_idx + chosen_neg
        order = torch.randperm(len(epoch_idx)).tolist()
        return iter([epoch_idx[o] for o in order])


def make_ratio_undersample_sampler(dataset: "BCSDBTDataset",
                                   pos_rate: float = 0.08) -> RandomRatioUndersampler:
    """목표 양성비(pos_rate) 무작위 언더샘플링 샘플러 팩토리 (음성 총량 축소 실험용)."""
    return RandomRatioUndersampler(dataset, pos_rate=pos_rate)


def build_dbt_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: Optional[int] = None,
    sampler: Optional[Sampler] = None,
    batch_sampler: Optional[Sampler] = None,
) -> DataLoader:
    """
    [RTX A6000 (48GB VRAM / 64-core 호스트) 최적화 DataLoader 팩토리]
    - batch_sampler 지정 시 batch_size/shuffle/sampler/drop_last 와 상호 배타
      (BalancedBatchSampler 처럼 배치 구성을 직접 제어할 때 사용).
    - sampler 지정 시 shuffle 은 무시(상호 배타).
    - num_workers: 호스트 64코어. 캐시 적중 시 .npz 로드가 가벼우므로 12~16 권장.
    - pin_memory=True: CPU→A6000 H2D 전송을 페이지락 메모리로 가속(.to(non_blocking=True)와 병행).
    - persistent_workers / prefetch_factor: 에폭 간 워커 재생성 비용 제거 및 선반입.
    - collate_fn: 가변 깊이 3D 볼륨 패딩 (collate_dbt_volumes).
    """
    if num_workers is None:
        num_workers = min(16, os.cpu_count() or 8)

    common = dict(
        num_workers=num_workers,
        collate_fn=collate_dbt_volumes,
        pin_memory=True,
    )
    if num_workers > 0:
        common['persistent_workers'] = True
        common['prefetch_factor'] = 4

    if batch_sampler is not None:
        # batch_sampler 는 batch_size/shuffle/sampler/drop_last 와 함께 줄 수 없음
        return DataLoader(dataset, batch_sampler=batch_sampler, **common)

    use_shuffle = shuffle if sampler is None else False
    drop_last = (sampler is not None) or shuffle  # 학습 배치는 마지막 불완전 배치 제거
    return DataLoader(dataset, batch_size=batch_size, shuffle=use_shuffle,
                      sampler=sampler, drop_last=drop_last, **common)


# --- (테스트 스니펫) ---
if __name__ == "__main__":
    print("[System] Initializing Data Pipeline Architect...")
    # 컨테이너 내부 데이터셋 마운트 경로
    # docker-compose: "/mnt/external_ssd/BreastCancer Datasets" -> /data/datasets (read-only)
    dataset_root = "/data/datasets/BCS-DBT"

    dataset = BCSDBTDataset(dataset_root=dataset_root, split='train')
    loader = build_dbt_dataloader(dataset, batch_size=2, shuffle=True, num_workers=2)
    batch = next(iter(loader))
    print(f"volumes: {tuple(batch['volumes'].shape)}, masks: {tuple(batch['masks'].shape)}, "
          f"depth_mask: {tuple(batch['depth_mask'].shape)}, labels: {batch['labels'].tolist()}")