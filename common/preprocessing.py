import cv2
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from typing import Tuple


class MedicalImagePreprocessor:
    """
    의료 영상(DICOM) 전처리 및 텐서 규격화 파이프라인
    Target: 2D FFDM & 3D DBT -> Swin Transformer (e.g., 384x384)

    [공용 모듈] fine_tuning(3D BCS-DBT) 과 transfer_learning(2D VinDr/CMMD/INbreast)
    양쪽에서 동일한 크롭→CLAHE→리사이즈/패딩 규격을 공유하기 위해 common 으로 분리.
    """

    def __init__(self, target_size: Tuple[int, int] = (384, 384)):
        self.target_size = target_size
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def read_volume(self, dicom_path: str) -> np.ndarray:
        """
        멀티프레임 DBT DICOM 1개 파일을 읽어 정규화된 uint8 볼륨 [Z, H, W] 로 반환.
        BCS-DBT 는 1개 뷰(=시리즈) 전체가 단일 멀티프레임 DICOM 으로 저장된다.
        단일프레임(2D FFDM: VinDr/CMMD/INbreast)이면 [1, H, W] 로 승격되어 동일 경로 처리.
        """
        dcm = pydicom.dcmread(dicom_path)
        arr = dcm.pixel_array  # 멀티프레임 → [Z, H, W], 단일프레임이면 [H, W]
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]

        # VOI LUT (윈도잉) 적용 — 멀티프레임 태그 누락 시 원본 유지
        try:
            arr = apply_voi_lut(arr, dcm)
        except Exception:
            pass

        # MONOCHROME1(반전 표시)을 MONOCHROME2 극성으로 통일
        if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = np.amax(arr) - arr

        # 볼륨 전체 기준 0~255 정규화 (슬라이스 간 밝기 일관성 유지)
        arr = arr.astype(np.float32)
        amin, amax = float(arr.min()), float(arr.max())
        denom = (amax - amin) if amax > amin else 1.0
        arr = (arr - amin) / denom * 255.0
        return arr.astype(np.uint8)

    def compute_crop_rect(self, volume_uint8: np.ndarray) -> Tuple[int, int, int, int]:
        """볼륨 전체 MIP(최대값 투영)에서 유방 영역 크롭 박스를 1회 산출 → 전 슬라이스 동일 적용."""
        mip = volume_uint8.max(axis=0)  # [H, W]
        return self._find_breast_crop_rect(mip)

    def finalize_image(self, frame_uint8: np.ndarray, crop_rect: Tuple[int, int, int, int]) -> np.ndarray:
        """크롭→CLAHE→리사이즈/패딩 → uint8 [th, tw] (캐시 저장에 적합한 형태)."""
        img = self._apply_crop(frame_uint8, crop_rect)
        img = self.clahe.apply(img)                 # 국소 대비 강화
        return self._pad_and_resize(img)            # 종횡비 보존 리사이즈+패딩 (uint8)

    def finalize_mask(self, mask_uint8: np.ndarray, crop_rect: Tuple[int, int, int, int]) -> np.ndarray:
        mask = self._apply_crop(mask_uint8, crop_rect)
        return self._pad_and_resize(mask, is_mask=True)  # uint8 [th, tw]

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
