"""
[통합 fine-tuning: BCS-DBT + DBT-2026] — train/val/test 3분할 + 안정화 스케줄.
- 데이터: BCS-DBT 공식 train/val/test + DBT-2026 train/val/test 를 split 별로 병합.
  train=학습, val=체크포인트 선택, test=최종 홀드아웃 리포트(선택에 미사용).
- 안정화(epoch-5 collapse 방지): 3-그룹 LLRD + 선형 warmup + 낮은 peak LR + grad-clip 0.5.
- 라벨: abnormal = benign-lesion + malignant (LABEL_DEFINITIONS.md). 기존 산출물 미수정,
  체크포인트 checkpoints/combined_*.pth.
실행: PYTHONPATH=/workspace python fine_tuning_combined/fine-tune_combined.py
"""
import os
import torch
import torch.nn.functional as F
import torch.amp as amp
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from common.model import DimensionAgnostic3DSwin
from fine_tuning.data_preprocessing import collate_dbt_volumes
from fine_tuning_combined.combined import build_combined_dataset, RatioUndersampler


def focal_loss_with_logits(logits, targets, alpha=0.75, gamma=2.0):
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * (1 - p_t).clamp(min=1e-6) ** gamma
    a_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (a_t * loss).mean()


def build_llrd_groups(model, lr_head, lr_low, lr_deep):
    """3-그룹 LLRD(차등 LR) — 사전학습 백본 보호(collapse 방지 핵심)."""
    head, low, deep = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith('head') or n.startswith('saliency_proj'):
            head.append(p)
        elif ('backbone.layers.2' in n or 'backbone.layers.3' in n or n.startswith('backbone.norm')):
            deep.append(p)
        else:
            low.append(p)
    return [{'params': head, 'lr': lr_head},
            {'params': low, 'lr': lr_low},
            {'params': deep, 'lr': lr_deep}]


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    """movavg8→max scan aggregation → AUROC/F1/Acc."""
    model.eval()
    all_labels, all_preds = [], []
    for batch in tqdm(loader, desc="[eval]", disable=None):
        volumes = batch['volumes'].to(device, non_blocking=True)
        depth_mask = batch['depth_mask'].to(device, non_blocking=True)
        labels = batch['labels']
        with amp.autocast('cuda', dtype=amp_dtype):
            logits, _ = model(volumes, depth_mask)
            probs = torch.sigmoid(logits.float())
        for b in range(volumes.size(0)):
            vz = int(depth_mask[b].sum().item())
            if vz < 8:
                all_preds.append(probs[b, :vz].max().item())
            else:
                vp = probs[b, :vz].unsqueeze(0).unsqueeze(0)
                sm = torch.nn.functional.avg_pool1d(vp, kernel_size=8, stride=1)
                all_preds.append(sm.max().item())
        all_labels.extend(labels.numpy())
    try:
        auroc = roc_auc_score(all_labels, all_preds)
        pb = (np.array(all_preds) > 0.5).astype(int)
        acc, f1 = accuracy_score(all_labels, pb), f1_score(all_labels, pb)
    except ValueError:
        auroc, acc, f1 = 0.0, 0.0, 0.0
    return auroc, f1, acc


def _seed_worker(wid):
    import random
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s); random.seed(s)


def _loader(ds, batch_size, num_workers, sampler=None, shuffle=False, generator=None):
    return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                      shuffle=(shuffle and sampler is None),
                      drop_last=(sampler is not None), num_workers=num_workers,
                      collate_fn=collate_dbt_volumes, pin_memory=True,
                      persistent_workers=(num_workers > 0),
                      prefetch_factor=(2 if num_workers > 0 else None),
                      generator=generator,
                      worker_init_fn=(_seed_worker if num_workers > 0 else None))


def train_combined(epochs=15, batch_size=8, num_workers=12, pos_rate=0.10,
                   neighbor_slices=1, patience=4, warmup_epochs=2,
                   lr_head=5e-5, lr_low=5e-6, lr_deep=1e-6, clip=0.5,
                   backbone='swin_base_patch4_window12_384', arch_tag='b',
                   cache_dir='/data/cache', ckpt_dir='checkpoints', pretrained=True, seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    gen = torch.Generator(); gen.manual_seed(seed)
    print(f"[Repro] seed={seed} (global RNG + DataLoader generator + worker seeding)")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    k = 2 * neighbor_slices + 1
    train_ds, train_flags = build_combined_dataset('train', cache_dir, neighbor_slices, augment=True)
    val_ds, _ = build_combined_dataset('val', cache_dir, neighbor_slices, augment=False)
    test_ds, _ = build_combined_dataset('test', cache_dir, neighbor_slices, augment=False)

    sampler = RatioUndersampler(train_flags, pos_rate=pos_rate)
    train_loader = _loader(train_ds, batch_size, num_workers, sampler=sampler, generator=gen)
    val_loader = _loader(val_ds, batch_size, num_workers, shuffle=False)
    test_loader = _loader(test_ds, batch_size, num_workers, shuffle=False)

    model = DimensionAgnostic3DSwin(pretrained=pretrained, freeze_backbone=False,
                                    in_chans=k, backbone=backbone).to(device)
    # [안정화] 3-그룹 LLRD + warmup→cosine + 낮은 peak LR
    optimizer = torch.optim.AdamW(build_llrd_groups(model, lr_head, lr_low, lr_deep),
                                  weight_decay=1e-2)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
    amp_dtype = torch.bfloat16
    print(f"[Stabilized] LLRD head={lr_head}/low={lr_low}/deep={lr_deep} | "
          f"warmup={warmup_epochs}ep | clip={clip}")

    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = f'{ckpt_dir}/combined_25d_swin{arch_tag}_r{neighbor_slices}_pos{int(round(pos_rate*100)):02d}_strat_pureN_b{batch_size}.pth'
    best_auroc, no_improve = 0.0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", disable=None):
            volumes = batch['volumes'].to(device, non_blocking=True)
            depth_mask = batch['depth_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast('cuda', dtype=amp_dtype):
                logits, _ = model(volumes, depth_mask)
                scan_logits, _ = logits.max(dim=1)
                loss = focal_loss_with_logits(scan_logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            optimizer.step()
        scheduler.step()

        auroc, f1, acc = evaluate(model, val_loader, device, amp_dtype)
        print(f"--> [Combined][VAL] Epoch {epoch} AUROC: {auroc:.4f} | F1: {f1:.4f} | Acc: {acc:.4f}")
        if auroc > best_auroc:
            best_auroc, no_improve = auroc, 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"[*] Best saved → {ckpt_path} (VAL AUROC {best_auroc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[EarlyStopping] stop at epoch {epoch} (best VAL {best_auroc:.4f})")
                break

    # ---- 최종 홀드아웃 TEST (best 체크포인트 로드, 선택에 미사용) ----
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    t_auroc, t_f1, t_acc = evaluate(model, test_loader, device, amp_dtype)
    print(f"==> [Combined][TEST] best-val ckpt → AUROC: {t_auroc:.4f} | F1: {t_f1:.4f} | Acc: {t_acc:.4f}")
    print(f"[Combined] done. best VAL={best_auroc:.4f} | held-out TEST={t_auroc:.4f} → {ckpt_path}")
    return best_auroc, t_auroc


if __name__ == "__main__":
    # BCS-DBT 공식 train/val/test + DBT-2026 를 split 별 병합, 10:90 언더샘플, Swin-B, 안정화.
    train_combined(epochs=15, batch_size=16, pos_rate=0.10, neighbor_slices=1,
                   num_workers=8,                # 12→8: batch16 host-RAM 안전
                   backbone='swin_base_patch4_window12_384', arch_tag='b',
                   ckpt_dir='checkpoints_b16')   # 새 체크포인트 공간 + batch 16
