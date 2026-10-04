import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp as amp
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

# 분리된 아키텍처 및 데이터 파이프라인 모듈 호출 (패키지 절대경로; PYTHONPATH=/workspace)
from fine_tuning.data_preprocessing import (
    BCSDBTDataset, build_dbt_dataloader, make_balanced_batch_sampler,
    make_ratio_undersample_sampler)
from common.model import DimensionAgnostic3DSwin


# ==============================================================================
# Loss functions
# ==============================================================================
def focal_loss_with_logits(logits, targets, alpha=0.75, gamma=2.0):
    """[분류] Binary Focal Loss. alpha=0.75 로 희소 양성(Cancer)을 UP-weight
    (alpha=0.25 는 양성을 다운웨이트 → 전부 음성 붕괴; 1:3 배치에 0.75 가 균형)."""
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * (1 - p_t).clamp(min=1e-6) ** gamma
    if alpha is not None:
        a_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = a_t * loss
    return loss.mean()


def dice_loss(pred, target, eps=1.0):
    """[공간 정렬] Dice Loss — 병변 박스가 슬라이스의 ~2%뿐이라 BCE 대신 사용
    (BCE 는 배경에 압도되어 saliency→0 으로 붕괴). pred/target: [K, H, W] ∈ [0,1]."""
    pred = pred.flatten(1)
    target = target.flatten(1)
    num = 2.0 * (pred * target).sum(dim=1) + eps
    den = pred.sum(dim=1) + target.sum(dim=1) + eps
    return (1.0 - num / den).mean()


def saliency_to_box(saliency_2d, thresh=0.5):
    """[추론 후처리] saliency map(384x384, [0,1]) → (x,y,w,h) bbox in 384-padded space.
    thresh=None 이면 Otsu. 박스 없으면 None."""
    sal = saliency_2d.detach().float().cpu().numpy()
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    if thresh is None:
        binm = cv2.threshold((sal * 255).astype(np.uint8), 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    else:
        binm = (sal > thresh).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return cv2.boundingRect(max(cnts, key=cv2.contourArea))  # (x, y, w, h)


def build_llrd_param_groups(model, lr_head, lr_low, lr_deep):
    """[LLRD] 계층별 차등 학습률.
    - head/saliency(신규): lr_head (높게)
    - backbone stage 0-1 + patch_embed(저수준 질감): lr_low
    - backbone stage 2-3 + norm(SSL 고수준 의미): lr_deep (낮게 → SSL 지식 보존)
    """
    head, low, deep = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith('head') or n.startswith('saliency_proj'):
            head.append(p)
        elif ('backbone.layers.2' in n or 'backbone.layers.3' in n
              or n.startswith('backbone.norm')):
            deep.append(p)
        else:  # backbone.patch_embed, backbone.layers.0/1
            low.append(p)
    print(f"[LLRD] head={sum(p.numel() for p in head)/1e6:.2f}M@{lr_head} "
          f"low={sum(p.numel() for p in low)/1e6:.2f}M@{lr_low} "
          f"deep={sum(p.numel() for p in deep)/1e6:.2f}M@{lr_deep}")
    return [
        {'params': head, 'lr': lr_head},
        {'params': low,  'lr': lr_low},
        {'params': deep, 'lr': lr_deep},
    ]


# ==============================================================================
# MLOps Fine-Tuning Engine
# ==============================================================================
def train_model(dataset_root: str, cache_dir: str, epochs: int = 15, batch_size: int = 8,
                num_workers: int = 12,
                lambda_att: float = 1.0, patience: int = 4, ssl_weights: str = None,
                freeze_backbone: bool = True,
                lr_head: float = 1e-4, lr_low: float = 1e-5, lr_deep: float = 1e-6,
                neighbor_slices: int = 1, pos_rate: float = 0.08,
                backbone: str = 'swin_base_patch4_window12_384', arch_tag: str = 'b'):
    print("\n[System] Booting Phase A (Supervised MIL + WSOL) Pipeline...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # [A6000 연산 최적화] Ampere TF32 행렬연산 + cuDNN 오토튜너 활성화
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    # 1. Dataloaders 초기화 (train 은 온더플라이 증강 ON → 반복 양성 과적합 완화)
    train_dataset = BCSDBTDataset(dataset_root, split='train', cache_dir=cache_dir, augment=True,
                                  neighbor_slices=neighbor_slices)
    val_dataset = BCSDBTDataset(dataset_root, split='val', cache_dir=cache_dir, augment=False,
                                neighbor_slices=neighbor_slices)

    # [불균형 해소 — 실험: 데이터셋 수준 무작위 언더샘플링] 음성 총량을 줄여 양성비를
    # pos_rate(기본 8%)로 맞춘다. 배치별 양성 보장이 없는 순수 무작위 다운샘플링이며,
    # 잔여 불균형은 Focal Loss(alpha=0.75)로 보정. '음성 축소'가 성능에 미치는 영향 관찰용.
    train_sampler = make_ratio_undersample_sampler(train_dataset, pos_rate=pos_rate)
    train_loader = build_dbt_dataloader(train_dataset, batch_size=batch_size,
                                        sampler=train_sampler, num_workers=num_workers)
    val_loader = build_dbt_dataloader(val_dataset, batch_size=batch_size,
                                      shuffle=False, num_workers=num_workers)

    # 2. Model / Optimizer
    # ssl_weights 지정 시 SSL 백본으로 초기화; freeze_backbone=False → 전체 개방 + LLRD.
    # [2.5D] in_chans = 2r+1 (인접 슬라이스 채널). r=0 → 기존 1채널과 동일.
    model = DimensionAgnostic3DSwin(pretrained=(ssl_weights is None),
                                    ssl_weights=ssl_weights,
                                    freeze_backbone=freeze_backbone,
                                    in_chans=2 * neighbor_slices + 1,
                                    backbone=backbone).to(device)
    if freeze_backbone:
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=lr_head, weight_decay=1e-2)
    else:
        # LLRD: 계층별 차등 학습률로 SSL 백본 미세 적응(catastrophic forgetting 방지)
        optimizer = torch.optim.AdamW(build_llrd_param_groups(model, lr_head, lr_low, lr_deep),
                                      weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp_dtype = torch.bfloat16  # Ampere 네이티브 bf16 (GradScaler 불필요)

    best_auroc = 0.0
    epochs_no_improve = 0
    os.makedirs('checkpoints', exist_ok=True)
    # 2.5D 실험은 in_chans 가 달라 기존 1채널 체크포인트와 호환되지 않음 → 새 이름으로 보존.
    # pos_rate 도 파일명에 포함해 다운샘플링 스윕 간 체크포인트 덮어쓰기 방지
    # (기존 8% 런의 best_25d_r1.pth[AUROC 0.7392] 보존).
    ckpt_path = f'checkpoints/best_25d_swin{arch_tag}_r{neighbor_slices}_pos{int(round(pos_rate * 100)):02d}.pth'

    for epoch in range(1, epochs + 1):
        # ------------------- TRAIN -------------------
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")

        for batch in pbar:
            volumes = batch['volumes'].to(device, non_blocking=True)
            depth_mask = batch['depth_mask'].to(device, non_blocking=True)
            masks = batch['masks'].to(device, non_blocking=True)[:, 0]  # [B, Z, H, W]
            labels = batch['labels'].to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            with amp.autocast('cuda', dtype=amp_dtype):
                logits, saliency = model(volumes, depth_mask)  # [B,Z], [B,Z,H,W]

                # (a) 분류: MIL max-pool → Focal Loss
                scan_logits, _ = logits.max(dim=1)
                loss_cls = focal_loss_with_logits(scan_logits, labels)

                # (b) 공간 정렬(WSOL): GT 박스가 있는 슬라이스에만 게이팅하여 Dice
                box_slices = masks.flatten(2).sum(dim=2) > 0          # [B, Z]
                if box_slices.any():
                    sal_sel = saliency[box_slices].float()            # [K, H, W]
                    msk_sel = masks[box_slices].float()               # [K, H, W]
                    loss_spatial = dice_loss(sal_sel, msk_sel)
                else:
                    loss_spatial = scan_logits.new_zeros(())

                loss = loss_cls + lambda_att * loss_spatial

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'cls': f"{loss_cls.item():.3f}",
                              'dice': f"{loss_spatial.item():.3f}"})

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # ----------------- VALIDATION -----------------
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []

        print(f"\n[System] Evaluating Phase 1 Validation (Epoch {epoch})...")
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                volumes = batch['volumes'].to(device, non_blocking=True)
                depth_mask = batch['depth_mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True).float()

                with amp.autocast('cuda', dtype=amp_dtype):
                    logits, _saliency = model(volumes, depth_mask)  # [B,Z], [B,Z,H,W]
                    scan_logits, _ = logits.max(dim=1)
                    loss = focal_loss_with_logits(scan_logits, labels)

                val_loss += loss.item()

                # [수학적 동기화: Kassis_2024 Section 1.1 지표 산출 파이프라인]
                # 패딩=-inf → sigmoid=0. avg_pool1d 안정성 위해 fp32 로 산출.
                probs = torch.sigmoid(logits.float())  # Step 1: p_z 산출 [B, Z]

                scan_probs = []
                for b in range(volumes.size(0)):
                    valid_z = depth_mask[b].sum().item()  # 실제 슬라이스 개수

                    if valid_z < 8:
                        # 슬라이스가 8장 미만인 예외 케이스는 단순 Max 처리
                        scan_probs.append(probs[b, :valid_z].max().item())
                        continue

                    # 유효 슬라이스 텐서화 [1, 1, Z] (Conv1d/AvgPool1d 입력 규격)
                    valid_probs = probs[b, :valid_z].unsqueeze(0).unsqueeze(0)

                    # Step 2: Moving Average Filter (W=8 적용)
                    # S_z = 1/W \sum p_{z+k}
                    smoothed_probs = torch.nn.functional.avg_pool1d(valid_probs, kernel_size=8, stride=1)

                    # Step 3: Scan-level Probability 도출 (최댓값)
                    # P_{scan} = max(S_z)
                    p_scan = smoothed_probs.max().item()
                    scan_probs.append(p_scan)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(scan_probs)  # 이동 평균 필터가 적용된 최종 확률값

        # 매트릭스 채점
        avg_val_loss = val_loss / len(val_loader)
        try:
            auroc = roc_auc_score(all_labels, all_preds)
            preds_binary = (np.array(all_preds) > 0.5).astype(int)
            acc = accuracy_score(all_labels, preds_binary)
            f1 = f1_score(all_labels, preds_binary)
        except ValueError:
            # 배치에 하나의 클래스만 있는 경우 에러 방지
            auroc, acc, f1 = 0.0, 0.0, 0.0

        print(f"--> [Result] Loss: {avg_val_loss:.4f} | AUROC: {auroc:.4f} | F1: {f1:.4f} | Acc: {acc:.4f}")

        # 최고 성능(Best Checkpoint) 저장 + 조기 종료(Early Stopping)
        if auroc > best_auroc:
            best_auroc = auroc
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"[*] Best Model Saved → {ckpt_path} (AUROC: {best_auroc:.4f})")
        else:
            epochs_no_improve += 1
            print(f"[EarlyStopping] no AUROC improvement for {epochs_no_improve}/{patience} epoch(s)")
            if epochs_no_improve >= patience:
                print(f"[EarlyStopping] stop at epoch {epoch} (best AUROC {best_auroc:.4f}).")
                break

    print("\n[System] Supervised fine-tuning complete: 2.5D MIL classifier "
          "(focal + auxiliary Dice) with LLRD on a MoCo-v2 self-supervised pretrained "
          "Swin-B backbone. Best checkpoint saved for downstream transfer learning.")


if __name__ == "__main__":
    # 데이터 경로 세팅 (실제 환경에 맞게 수정)
    DATASET_ROOT = "/data/datasets/BCS-DBT"  # 도커 마운트 경로
    # 프리워밍된 전처리 캐시 위치와 반드시 일치해야 함(/data/cache/preproc/<split>).
    # 잘못 지정 시 캐시 미적중 → 전 볼륨 재디코딩(~8시간).
    CACHE_DIR = "/data/cache"

    # === Ceiling-Break Test ===: SSL 백본 + 전체 개방(LLRD) + 1:3 Focal(a=0.75) + Dice(λ=0.1)
    # LLRD: head 1e-4 / stage0-1 1e-5 / stage2-3(SSL 의미) 1e-6. baseline to beat = Phase A 0.732.
    # Swin-Small(48.8M, ImageNet-init, SSL 비호환) + 2.5D k=3 + 5:95.
    # 목적: 파라미터 축소로 8%/5% Swin-B 런의 과적합 완화 검증.
    train_model(dataset_root=DATASET_ROOT, cache_dir=CACHE_DIR, epochs=15, batch_size=8,
                lambda_att=0.1, patience=4,
                ssl_weights=None,    # Swin-S: SSL 백본(Swin-B 전용) 사용 불가 → ImageNet init
                freeze_backbone=False, lr_head=1e-4, lr_low=1e-5, lr_deep=1e-6,
                neighbor_slices=1,   # 2.5D: k=3 (±1 인접 슬라이스)
                pos_rate=0.05,       # 무작위 언더샘플링: 양성 5% / 음성 95% (neg 1444)
                backbone='swin_small_patch4_window7_224', arch_tag='s')