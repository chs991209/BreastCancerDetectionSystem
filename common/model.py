import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional
import timm


class CosineHead(nn.Module):
    """코사인 유사도 분류 헤드(binary): logit = scale · cos(feature, w). 단일 로짓 → MIL/파이프라인 호환."""
    def __init__(self, dim, scale: float = 16.0):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1, dim))
        self.scale = scale

    def forward(self, x):                      # x: [n, dim]
        xf = F.normalize(x, dim=1)
        wf = F.normalize(self.w, dim=1)
        return self.scale * (xf @ wf.t())      # [n, 1]


class DimensionAgnostic3DSwin(nn.Module):
    """
    [Architectural Defense]
    RTX A6000/4090 방어를 위한 Z축 슬라이싱 2D-to-3D 하이브리드 아키텍처
    """

    def __init__(self, pretrained: bool = True, slice_chunk: int = 32,
                 ssl_weights: Optional[str] = None, freeze_backbone: bool = True,
                 in_chans: int = 1,
                 backbone: str = 'swin_base_patch4_window12_384', img_size: int = 384,
                 head_type: str = 'mlp'):
        super().__init__()
        # 백본에 한 번에 통과시킬 슬라이스 묶음 크기(VRAM-처리량 트레이드오프).
        # Z 루프를 [N_slice] 벡터 배치로 묶어 A6000 행렬연산 활용도를 극대화한다.
        self.slice_chunk = slice_chunk

        # [2.5D] 입력 채널 수 k(=2r+1). DBT 는 인접 슬라이스 스택, 2D FFDM 은 k=1 복제.
        # k=1 이면 기존(흑백 단일채널) 동작과 완전 동일.
        self.in_chans = in_chans
        self.backbone_name = backbone

        # ssl_weights 지정 시 ImageNet 다운로드 생략(SSL 가중치로 초기화).
        # 주의: SSL 백본(ssl_swinb_backbone.pth)은 Swin-B 전용 → Swin-S 등 다른 백본과 비호환.
        if ssl_weights:
            pretrained = False

        # 백본 로드 (in_chans=k). timm 이 ImageNet(3ch)→k ch patch_embed 를 자동 적응.
        # Swin-S 는 native 384 변형이 없어 window7_224 에 img_size=384 오버라이드로 사용.
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,   # Classifier 제거, Feature만 추출
            img_size=img_size,
        )
        self.feature_dim = self.backbone.num_features

        # [Phase B → A] SIFT-DBT SSL 사전학습 백본 로드 (pretrain_ssl.py 산출물).
        # 키가 동일하므로 strict 로드(검증 완료: 327 keys identical).
        if ssl_weights:
            state = torch.load(ssl_weights, map_location='cpu')
            # SSL 백본은 in_chans=1 로 학습됨. 2.5D(k>1)면 patch_embed 를 '팽창(inflation)'
            # — 1채널 커널을 k 개로 복제 후 1/k 스케일(활성값 크기 보존). deflation 의 역연산이며,
            # 죽은 채널 문제(depth-mean=0)가 없다(복제라 채널 간 동일 → 이후 학습으로 분화).
            pe_key = 'patch_embed.proj.weight'
            if in_chans > 1 and pe_key in state and state[pe_key].shape[1] == 1:
                w = state[pe_key]                                    # [C, 1, p, p]
                state[pe_key] = w.repeat(1, in_chans, 1, 1) / in_chans  # [C, k, p, p]
                print(f"[Architecture] Inflated SSL patch_embed 1->{in_chans}ch (replicate / k)")
            self.backbone.load_state_dict(state, strict=True)
            print(f"[Architecture] Loaded SSL-pretrained backbone from {ssl_weights}")

        # Binary Classification Head (Cancer vs Benign)
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim // 2, 1)
        ) if head_type == 'mlp' else CosineHead(self.feature_dim)

        # [WSOL] 최종 feature map(C채널)을 1채널 saliency 로 사영하는 1x1 Conv.
        # forward_features 출력(채널-라스트 [N,h,w,C])을 permute 후 통과시킨다.
        self.saliency_proj = nn.Conv2d(self.feature_dim, 1, kernel_size=1)

        if freeze_backbone:
            # Phase A: ImageNet 백본 → 상위 stage 동결(과적합·VRAM 억제)
            self._apply_orthogonal_freeze()
        else:
            # Phase B fine-tune: SSL 백본 전체 개방(LLRD 로 보호). 전부 학습.
            for p in self.parameters():
                p.requires_grad = True
            tunable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"[Architecture] FULL backbone trainable (LLRD): {tunable / 1e6:.2f}M params")

    def _apply_orthogonal_freeze(self):
        """
        [직교적 파라미터 동결 (Orthogonal Parameter Separation)]
        글로벌 위상을 담당하는 상위 레이어(Stage 2, 3)를 동결(80%)하고,
        질감을 담당하는 하위 레이어(Stage 0, 1)와 Head만 튜닝(20%)하여
        VRAM 소모를 줄이고 과적합을 방지합니다.
        """
        # 모든 파라미터 1차 동결
        for param in self.parameters():
            param.requires_grad = False

        # 하위 레이어 (Local Texture) 개방
        for name, param in self.backbone.named_parameters():
            if "layers.0" in name or "layers.1" in name or "patch_embed" in name:
                param.requires_grad = True

        # 분류 헤드 + saliency 사영층 개방
        for param in self.head.parameters():
            param.requires_grad = True
        for param in self.saliency_proj.parameters():
            param.requires_grad = True

        tunable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"[Architecture] Tunable params: {tunable / 1e6:.2f}M | Frozen params: {frozen / 1e6:.2f}M")

    def _feature_map(self, chunk):
        """백본의 최종 spatial feature map 추출 → 채널-퍼스트 [N, C, h, w]."""
        fmap = self.backbone.forward_features(chunk)   # timm Swin: 보통 [N, h, w, C]
        if fmap.ndim == 3:                             # [N, L, C] 형태면 정사각 복원
            n, l, c = fmap.shape
            s = int(round(l ** 0.5))
            fmap = fmap.reshape(n, s, s, c)
        # Fix 1: 채널-라스트 [N,h,w,C] → 채널-퍼스트 [N,C,h,w]
        return fmap.permute(0, 3, 1, 2).contiguous()

    def forward(self, volumes, depth_mask):
        """
        volumes: [B, 1, Z, H, W]
        depth_mask: [B, Z] (True: 실제 데이터, False: 패딩)
        반환:
            logits        : [B, Z]            (슬라이스별 로짓, 패딩 = -inf)
            saliency_maps : [B, Z, H, W]      (슬라이스별 CAM ∈ [0,1], 패딩 = 0)

        [WSOL] 최종 feature map → 1x1 Conv(1채널) → 384x384 업샘플 → sigmoid 로
        픽셀 saliency 를 생성한다. 분류 로짓은 동일 feature map 의 global-avg-pool 에서.
        성능: 유효 슬라이스만 모아 slice_chunk 묶음으로 백본 1회 통과(학습 시 checkpoint).
        """
        B, C, Z, H, W = volumes.shape

        x = volumes.permute(0, 2, 1, 3, 4).reshape(B * Z, C, H, W)
        flat_mask = depth_mask.reshape(B * Z)
        valid_idx = flat_mask.nonzero(as_tuple=True)[0]          # 실제 슬라이스 위치
        valid_x = x.index_select(0, valid_idx)                   # [N_valid, C, H, W]

        logit_chunks, sal_chunks = [], []
        for i in range(0, valid_x.size(0), self.slice_chunk):
            chunk = valid_x[i:i + self.slice_chunk]
            if self.training:
                fmap = checkpoint(self._feature_map, chunk, use_reentrant=False)  # [n,C,h,w]
            else:
                fmap = self._feature_map(chunk)

            pooled = fmap.mean(dim=(2, 3))               # [n, C]  global avg pool → 분류
            logit_chunks.append(self.head(pooled))       # [n, 1]

            sal = self.saliency_proj(fmap)               # [n, 1, h, w]
            sal = F.interpolate(sal, size=(H, W), mode='bilinear', align_corners=False)
            sal_chunks.append(torch.sigmoid(sal).squeeze(1))   # [n, H, W]

        valid_logits = torch.cat(logit_chunks, dim=0).squeeze(1)   # [N_valid]
        valid_sal = torch.cat(sal_chunks, dim=0)                   # [N_valid, H, W]

        # 로짓 캔버스: 패딩 = -inf
        logits = torch.full((B * Z,), float('-inf'),
                            device=volumes.device, dtype=valid_logits.dtype)
        logits = logits.index_copy(0, valid_idx, valid_logits).view(B, Z)

        # saliency 캔버스: 패딩 = 0
        sal_full = torch.zeros((B * Z, H, W), device=volumes.device, dtype=valid_sal.dtype)
        sal_full = sal_full.index_copy(0, valid_idx, valid_sal).view(B, Z, H, W)

        return logits, sal_full