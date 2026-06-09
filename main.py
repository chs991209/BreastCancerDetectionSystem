import os
import cv2
import json
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import torch
from torch.utils.data import Dataset
from typing import Union, Tuple, List, Dict, Optional
from tqdm import tqdm


class MedicalImagePreprocessor:
    """
    의료 영상(DICOM) 전처리 및 텐서 규격화 파이프라인
    Target: 2D FFDM & 3D DBT -> Swin Transformer (e.g., 384x384)
    """

    def __init__(self, target_size: Tuple[int, int] = (384, 384)):
        self.target_size = target_size
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def process_2d_dicom_with_mask(self, dicom_path: str, box_coords: Optional[Tuple[int, int, int, int]] = None) -> \
    Tuple[torch.Tensor, torch.Tensor]:
        # 1. 원본 디코딩
        img = self._read_and_normalize_dicom(dicom_path)

        # 2. 정답 마스크(M_box) 캔버스 초기화 및 바운딩 박스 렌더링
        mask = np.zeros_like(img, dtype=np.uint8)
        if box_coords is not None:
            bx, by, bw, bh = [int(v) for v in box_coords]
            cv2.rectangle(mask, (bx, by), (bx + bw, by + bh), 255, -1)

        # 3. 유방 조직 객체 탐지 및 이미지/마스크 '동시' 크롭 (좌표계 동기화)
        crop_rect = self._find_breast_crop_rect(img)
        img = self._apply_crop(img, crop_rect)
        mask = self._apply_crop(mask, crop_rect)

        # 4. 국소적 대비 강화 (마스크는 적용 제외)
        img = self.clahe.apply(img)

        # 5. 종횡비 보존 리사이즈 및 패딩 '동시' 적용
        img = self._pad_and_resize(img)
        mask = self._pad_and_resize(mask, is_mask=True)

        # 6. Tensor 변환 및 스케일링
        tensor_img = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        tensor_mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0

        return tensor_img, tensor_mask

    def _read_and_normalize_dicom(self, dicom_path: str) -> np.ndarray:
        dcm = pydicom.dcmread(dicom_path)
        pixel_array = apply_voi_lut(dcm.pixel_array, dcm)
        if dcm.PhotometricInterpretation == "MONOCHROME1":
            pixel_array = np.amax(pixel_array) - pixel_array
        pixel_array = pixel_array - np.min(pixel_array)
        pixel_array = (pixel_array / np.max(pixel_array)) * 255.0
        return pixel_array.astype(np.uint8)

    def _find_breast_crop_rect(self, img: np.ndarray) -> Tuple[int, int, int, int]:
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return (0, 0, img.shape[1], img.shape[0])
        c = max(contours, key=cv2.contourArea)
        return cv2.boundingRect(c)

    def _apply_crop(self, img: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = rect
        return img[y:y + h, x:x + w]

    def _pad_and_resize(self, img: np.ndarray, is_mask: bool = False) -> np.ndarray:
        h, w = img.shape[:2]
        target_h, target_w = self.target_size
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
        resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
        delta_w, delta_h = target_w - new_w, target_h - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])


class BCSDBTDataset(Dataset):
    """
    [Architecture Update: System IO Optimization & Meta-data Alignment]
    BCS-DBT의 복잡한 매니페스트와 CSV 구조를 해석하여 3D 텐서 볼륨을 동적으로 조립합니다.
    """

    def __init__(self, root_dir: str, split: str = 'train', target_size: Tuple[int, int] = (384, 384)):
        self.root_dir = root_dir
        self.split = split
        self.dicom_dir = os.path.join(root_dir, 'breast_cancer_screening_dbt')
        self.csv_dir = os.path.join(root_dir, 'Original')

        self.preprocessor = MedicalImagePreprocessor(target_size=target_size)

        # 1. 파일 경로 및 메타데이터 초기화
        self._load_metadata()

    def _load_metadata(self):
        print(f"[System] Building Data Index for {self.split.upper()} split...")

        # 매니페스트 로드 (File -> Patient ID, Series ID 매핑)
        # 예시 헤더: File Instance Name,Collection ID,Patient ID,Study ID,Series ID
        manifest_files = [f for f in os.listdir(self.root_dir) if f.startswith('idc_file_id_manifest')]
        manifest_path = os.path.join(self.root_dir, manifest_files[0])
        self.manifest_df = pd.read_csv(manifest_path)

        # 라벨 및 박스 CSV 로드
        if self.split == 'train':
            self.boxes_df = pd.read_csv(os.path.join(self.csv_dir, 'BCS-DBT-boxes-train-v2.csv'))
            self.labels_df = pd.read_csv(os.path.join(self.csv_dir, 'BCS-DBT-labels-train-v2.csv'))
        else:
            self.boxes_df = pd.read_csv(os.path.join(self.csv_dir, 'BCS-DBT-boxes-validation-v2-PHASE-2-Jan-2024.csv'))
            self.labels_df = pd.read_csv(os.path.join(self.csv_dir, 'BCS-DBT-labels-validation-PHASE-2-Jan-2024.csv'))

        # 2. I/O 병목 방지를 위한 Series -> View (lcc, rmlo) 캐시 빌드
        self.series_to_view_cache = self._build_series_view_cache()

        # 3. 유효한 3D 볼륨(Series) 리스트업
        self.valid_volumes = []
        grouped_manifest = self.manifest_df.groupby('Series ID')

        for series_id, group in grouped_manifest:
            patient_id = group.iloc[0]['Patient ID']
            view = self.series_to_view_cache.get(series_id)

            if not view: continue  # 캐시에 없거나 에러난 시리즈 스킵

            # 현재 Split의 Labels CSV에 이 환자+뷰가 존재하는지 확인
            match = self.labels_df[(self.labels_df['PatientID'] == patient_id) & (self.labels_df['View'] == view)]
            if not match.empty:
                dicom_files = group['File Instance Name'].tolist()
                self.valid_volumes.append({
                    'patient_id': patient_id,
                    'series_id': series_id,
                    'view': view,
                    'files': dicom_files
                })
        print(f"[System] Successfully indexed {len(self.valid_volumes)} 3D Volumes.")

    def _build_series_view_cache(self) -> Dict[str, str]:
        """
        [엔지니어링 코어: I/O 최적화]
        모든 DICOM을 읽는 것은 재앙입니다. 각 Series 당 딱 1장의 DICOM 헤더만 읽어
        해당 볼륨이 'lcc'인지 'rmlo'인지 알아내고 JSON 파일로 영구 캐싱합니다.
        """
        cache_path = os.path.join(self.root_dir, 'series_view_cache.json')
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)

        print("[System] Generating Series-to-View Index Cache (Run Once)...")
        cache = {}
        for series_id, group in tqdm(self.manifest_df.groupby('Series ID')):
            first_file = group.iloc[0]['File Instance Name']
            file_path = os.path.join(self.dicom_dir, first_file)
            try:
                # stop_before_pixels 옵션으로 헤더만 0.01초만에 초고속 리딩
                dcm = pydicom.dcmread(file_path, stop_before_pixels=True)
                laterality = getattr(dcm, 'ImageLaterality', '').lower()
                view_pos = getattr(dcm, 'ViewPosition', '').lower()

                if laterality and view_pos:
                    cache[series_id] = laterality + view_pos  # e.g., 'l' + 'cc' -> 'lcc'
            except Exception:
                continue

        with open(cache_path, 'w') as f:
            json.dump(cache, f)
        return cache

    def __len__(self):
        return len(self.valid_volumes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        vol_info = self.valid_volumes[idx]
        patient_id = vol_info['patient_id']
        view = vol_info['view']
        file_names = vol_info['files']

        # 1. 파일들을 InstanceNumber(Z축 깊이) 기준으로 정렬
        dicom_slices = []
        for fname in file_names:
            fpath = os.path.join(self.dicom_dir, fname)
            dcm = pydicom.dcmread(fpath, stop_before_pixels=True)
            instance_num = int(dcm.InstanceNumber) if 'InstanceNumber' in dcm else 0
            dicom_slices.append((instance_num, fpath))

        dicom_slices.sort(key=lambda x: x[0])
        sorted_paths = [path for _, path in dicom_slices]

        # 2. 이 볼륨(Patient + View)에 해당하는 Bounding Boxes 추출
        boxes_subset = self.boxes_df[(self.boxes_df['PatientID'] == patient_id) & (self.boxes_df['View'] == view)]
        slice_to_boxes = {}
        for _, row in boxes_subset.iterrows():
            z_idx = int(row['Slice'])  # CSV의 Slice 번호 (0-indexed)
            if z_idx not in slice_to_boxes:
                slice_to_boxes[z_idx] = []
            slice_to_boxes[z_idx].append((row['X'], row['Y'], row['Width'], row['Height']))

        # 3. Z축 순회하며 볼륨 조립
        processed_images = []
        processed_masks = []

        for z_idx, path in enumerate(sorted_paths):
            # 이 슬라이스에 박스가 여러 개면 가장 첫 번째 것 사용 (단순화)
            box_coords = slice_to_boxes[z_idx][0] if z_idx in slice_to_boxes else None

            tensor_img, tensor_mask = self.preprocessor.process_2d_dicom_with_mask(path, box_coords)
            processed_images.append(tensor_img)
            processed_masks.append(tensor_mask)

        volume_tensor = torch.stack(processed_images, dim=1)  # Shape: [1, Z, H, W]
        mask_volume = torch.stack(processed_masks, dim=1)  # Shape: [1, Z, H, W]

        # 라벨 추출 (0: 정상/양성, 1: 악성/Cancer)
        label_row = self.labels_df[(self.labels_df['PatientID'] == patient_id) & (self.labels_df['View'] == view)].iloc[
            0]
        cancer_label = int(label_row['Cancer'])

        return volume_tensor, mask_volume, cancer_label


# --- (테스트 스니펫) ---
if __name__ == "__main__":
    print("[System] Initializing Data Pipeline Architect...")
    # 프로젝트 루트 경로 (사용자 환경 맞춤)
    root_directory = "./data/bc_data_3D/BCS-DBT"

    # 디렉토리 구조가 준비되어 있다면 아래 코드로 DataLoader 연결 가능
    # dataset = BCSDBTDataset(root_dir=root_di
    # vol_tensor, mask_tensor, label = dataset[0]
    # print(f"Output Volume Shape: {vol_tensor.shape}, Mask Shape: {mask_tensor.shape}, Label: {label}")