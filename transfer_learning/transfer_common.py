"""
[2D Transfer-Learning 공용 학습/평가 루프]
데이터셋 비의존적. 각 dataset(run.py)에서 dataset_cls 를 주입해 호출한다.

DBT(3D 2.5D) → 2D 전이(방향 B): dbt_ckpt 에 fine_tuning 산출 체크포인트
(best_25d_r{r}.pth, in_chans=k 동일)를 주면 strict 로드로 '가중치 전이' 후 미세조정.
dbt_ckpt=None 이면 ImageNet 초기화(전이 없이 2D 단독 학습 — 베이스라인).
2D 는 Z=1 볼륨이라 MIL max 는 항등. WSOL/Dice 미사용(박스 없음), 분류(focal)만.
"""
import os
import torch
import torch.nn.functional as F
import torch.amp as amp
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Sampler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from common.model import DimensionAgnostic3DSwin
from transfer_learning.mammo2d_base import collate_mammo2d


class RatioUndersampler(Sampler):
    """[2D 클래스 균형] 매 epoch 양성 전량 + 음성 무작위 부분추출로 목표 양성비(pos_rate)를
    맞춘다. VinDr(5% 양성)의 all-negative 붕괴 방지용. records[i]['label'] 사용."""
    def __init__(self, dataset, pos_rate: float = 0.20):
        labels = [int(r['label']) for r in dataset.records]
        self.pos_idx = [i for i, y in enumerate(labels) if y == 1]
        self.neg_idx = [i for i, y in enumerate(labels) if y == 0]
        n_pos = len(self.pos_idx)
        n_neg_keep = int(round(n_pos * (1.0 - pos_rate) / pos_rate))
        self.n_neg_keep = min(n_neg_keep, len(self.neg_idx))
        self.epoch_len = n_pos + self.n_neg_keep
        print(f"[Sampler] ratio-undersample: pos={n_pos} + neg {self.n_neg_keep}/{len(self.neg_idx)}"
              f" → epoch={self.epoch_len}, pos_rate={100 * n_pos / self.epoch_len:.1f}%")

    def __len__(self):
        return self.epoch_len

    def __iter__(self):
        perm = torch.randperm(len(self.neg_idx))[:self.n_neg_keep]
        idx = self.pos_idx + [self.neg_idx[p] for p in perm.tolist()]
        return iter([idx[o] for o in torch.randperm(len(idx)).tolist()])


class BalancedUndersampler(Sampler):
    """[2D 대칭 균형] 다수 클래스를 무작위 다운샘플링하여 매 epoch 1:1(normal:abnormal).
    방향 무관 — VinDr/INbreast(정상 다수)는 정상을, CMMD(비정상 다수)는 비정상을 줄인다.
    balanced batches → all-majority 붕괴(예: VinDr all-negative) 원천 차단."""
    def __init__(self, dataset):
        labels = [int(r['label']) for r in dataset.records]
        self.pos_idx = [i for i, y in enumerate(labels) if y == 1]
        self.neg_idx = [i for i, y in enumerate(labels) if y == 0]
        self.k = min(len(self.pos_idx), len(self.neg_idx))   # 각 클래스에서 k개씩
        print(f"[Sampler] balanced-undersample: keep {self.k} pos + {self.k} neg "
              f"(from {len(self.pos_idx)} pos / {len(self.neg_idx)} neg) → epoch={2 * self.k}, 1:1")

    def __len__(self):
        return 2 * self.k

    def __iter__(self):
        p = [self.pos_idx[i] for i in torch.randperm(len(self.pos_idx))[:self.k].tolist()]
        n = [self.neg_idx[i] for i in torch.randperm(len(self.neg_idx))[:self.k].tolist()]
        idx = p + n
        return iter([idx[o] for o in torch.randperm(len(idx)).tolist()])


def focal_loss_with_logits(logits, targets, alpha=0.75, gamma=2.0):
    """fine_tuning 과 동일한 Binary Focal Loss (희소 양성 UP-weight)."""
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * (1 - p_t).clamp(min=1e-6) ** gamma
    if alpha is not None:
        a_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = a_t * loss
    return loss.mean()


def kd_bce_with_logits(student_logits, teacher_logits, T=2.0):
    """[교차차원 지식증류] 이진 KD: teacher 의 soft label(온도 T sigmoid)을 student 가 모방.
    T^2 스케일로 그래디언트 크기 보정(Hinton). 3D-DBT teacher → 2D student 로 구조 지식 전달."""
    soft_teacher = torch.sigmoid(teacher_logits.detach() / T)
    return F.binary_cross_entropy_with_logits(student_logits / T, soft_teacher) * (T * T)


def build_llrd_groups(model, base_lr: float, decay: float = 0.75, weight_decay: float = 1e-2,
                      lrs=None):
    """[층별 학습률 감쇠 (Layer-wise LR Decay, LLRD)]
    각 Swin stage 를 '별개의' optimizer param-group 으로 구성한다.
    - plain(기하 감쇠): lr = base_lr * decay^depth. head→stem 으로 갈수록 단조 감소.
    - imbalanced(불균형): lrs=[m0..m5] (depth 0..5 별 배수) 를 주면 lr = base_lr * lrs[depth].
      단조 감쇠가 아니라 '병변의 미세한 차이' 가설에 맞춰 특정 depth(고해상 fine stage)를
      의도적으로 크게/작게 배분(비단조 bump)할 수 있다.
    depth 순서: head/norm(0) → layers.3=stage4(1) → layers.2=stage3(2) → layers.1=stage2(3)
      → layers.0=stage1(4) → patch_embed=stem(5). 저수준 texture(fine)는 depth 3~4(고해상).
    requires_grad=False(동결) 파라미터는 제외하므로 train_stages(부분 미세조정)와 그대로 결합된다.
    bias / norm 류는 weight_decay=0 (관례)."""
    def lr_at(depth):
        return base_lr * (lrs[depth] if lrs is not None else decay ** depth)
    # (매칭 prefix, depth) — 더 구체적인 prefix 를 앞에 두어 먼저 매칭.
    depth_of = [
        ('backbone.patch_embed', 5),
        ('backbone.layers.0', 4),
        ('backbone.layers.1', 3),
        ('backbone.layers.2', 2),
        ('backbone.layers.3', 1),
        ('backbone.norm', 0),
        ('head', 0),
        ('saliency_proj', 0),
    ]
    groups = {}   # (depth, decay_flag) → {'params': [...], 'lr':..., 'weight_decay':...}
    n_by_depth = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        depth = next((d for pre, d in depth_of if n.startswith(pre)), 0)
        no_decay = (p.ndim == 1) or n.endswith('.bias')   # norm/bias → WD 0
        key = (depth, no_decay)
        g = groups.setdefault(key, {'params': [], 'lr': lr_at(depth),
                                    'weight_decay': 0.0 if no_decay else weight_decay})
        g['params'].append(p)
        n_by_depth[depth] = n_by_depth.get(depth, 0) + p.numel()
    mode = 'imbalanced' if lrs is not None else f'geometric(decay={decay})'
    print(f"[LLRD] mode={mode}")
    for d in sorted(n_by_depth):
        print(f"[LLRD] depth {d}: lr={lr_at(d):.2e} | {n_by_depth[d] / 1e6:.1f}M params")
    return list(groups.values())


@torch.no_grad()
def _eval_split(model, loader, device, amp_dtype, desc='Eval'):
    """공용 평가: (auroc, f1, acc, preds, labels). preds/labels 는 DeLong·부트스트랩 CI 용 원자료."""
    model.eval()
    all_labels, all_preds = [], []
    for batch in tqdm(loader, desc=desc, disable=None):
        volumes = batch['volumes'].to(device, non_blocking=True)
        depth_mask = batch['depth_mask'].to(device, non_blocking=True)
        with amp.autocast('cuda', dtype=amp_dtype):
            logits, _ = model(volumes, depth_mask)
            scan_logits, _ = logits.max(dim=1)
        all_preds.extend(torch.sigmoid(scan_logits.float()).cpu().numpy().tolist())
        all_labels.extend(batch['labels'].numpy().tolist())
    try:
        auroc = roc_auc_score(all_labels, all_preds)
        pb = (np.array(all_preds) > 0.5).astype(int)
        acc, f1 = accuracy_score(all_labels, pb), f1_score(all_labels, pb)
    except ValueError:
        auroc, acc, f1 = 0.0, 0.0, 0.0
    return auroc, f1, acc, all_preds, all_labels


def run_transfer(dataset_cls, name: str, neighbor_slices: int = 1,
                 dbt_ckpt: str = None, epochs: int = 10, batch_size: int = 16,
                 lr: float = 1e-4, num_workers: int = 12, patience: int = 4,
                 cache_dir: str = '/data/cache', pos_rate: float = None, balance: bool = False,
                 backbone: str = 'swin_base_patch4_window12_384', arch_tag: str = 'b',
                 teacher_ckpt: str = None, teacher_backbone: str = None,
                 kd_alpha: float = 0.5, kd_temp: float = 2.0, train_stages=None,
                 llrd: bool = False, llrd_decay: float = 0.75, llrd_lrs=None, seed: int = None,
                 feature_extract: bool = False, init_ckpt: str = None,
                 eval_test: bool = False, preds_out: str = None,
                 head_type: str = 'mlp', loss_type: str = 'focal'):
    # feature_extract=True → 백본 전체 동결, head 만 학습(가중치 미세조정 없는 '특징 추출기' 전이).
    #   train_stages=[] 와 동치. (LP-FT 의 Phase A 로도 사용.)
    if feature_extract and train_stages is None:
        train_stages = []
    # init_ckpt: dbt_ckpt 대신 임의 체크포인트로 초기화(LP-FT Phase B: Phase A 산출물 로드용).
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    # [재현성] seed 지정 시 전역 RNG + DataLoader generator + worker seeding 고정.
    # (합성 언더샘플러의 매-epoch 무작위 재추출을 끄는 것과 함께 논문 재현성 보장.)
    g = None
    if seed is not None:
        import random
        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        g = torch.Generator(); g.manual_seed(seed)
        print(f"[Repro] seed={seed} (global RNG + DataLoader generator + worker seeding)")

    def _worker_init(wid):
        s = (seed if seed is not None else 0) + wid
        np.random.seed(s % (2**32)); import random as _r; _r.seed(s)

    k = 2 * neighbor_slices + 1
    train_ds = dataset_cls(split='train', neighbor_slices=neighbor_slices,
                           augment=True, cache_dir=cache_dir)
    val_ds = dataset_cls(split='val', neighbor_slices=neighbor_slices,
                         augment=False, cache_dir=cache_dir)
    # balance=True → 대칭 1:1 무작위 다운샘플(다수 클래스 축소); pos_rate → 비율 언더샘플; 아니면 shuffle.
    if balance:
        sampler = BalancedUndersampler(train_ds)
    elif pos_rate:
        sampler = RatioUndersampler(train_ds, pos_rate=pos_rate)
    else:
        sampler = None
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=(sampler is None), sampler=sampler, drop_last=True,
                              num_workers=num_workers, collate_fn=collate_mammo2d,
                              pin_memory=True, persistent_workers=(num_workers > 0),
                              prefetch_factor=(4 if num_workers > 0 else None),
                              generator=g, worker_init_fn=(_worker_init if seed is not None else None))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_mammo2d,
                            pin_memory=True, persistent_workers=(num_workers > 0),
                            prefetch_factor=(4 if num_workers > 0 else None))

    kd_mode = teacher_ckpt is not None
    # KD 모드: student 는 ImageNet init 로 '자기 가중치'를 학습(구조 지식은 loss 로만 전달).
    # 가중치전이 모드: dbt_ckpt 를 strict 로드해 초기화.
    student_pretrained = True if kd_mode else (dbt_ckpt is None)
    model = DimensionAgnostic3DSwin(pretrained=student_pretrained,
                                    freeze_backbone=False, in_chans=k,
                                    backbone=backbone, head_type=head_type).to(device)
    load_from = init_ckpt or dbt_ckpt   # init_ckpt(LP-FT Phase B) 우선, 없으면 dbt_ckpt(가중치전이 B)
    if load_from and not kd_mode:
        state = torch.load(load_from, map_location='cpu')
        if head_type != 'mlp':
            # 코사인 헤드는 체크포인트의 MLP head 와 shape 불일치 → 백본만 전이(head 신규 학습).
            state = {k2: v for k2, v in state.items() if not k2.startswith('head.')}
            model.load_state_dict(state, strict=False)
        else:
            model.load_state_dict(state, strict=True)   # in_chans 동일 → strict OK
        tag = 'init_ckpt (LP-FT phase B)' if init_ckpt else 'DBT 2.5D weights (weight transfer B)'
        print(f"[Transfer] Loaded {tag} from {load_from}")

    # [교차차원 KD] frozen 3D-DBT teacher 로드 (자체 backbone). 2D 타깃 이미지에 대해
    # soft label 을 생성 → student 가 focal(hard) + KD(soft) 혼합으로 학습.
    teacher = None
    if kd_mode:
        teacher = DimensionAgnostic3DSwin(pretrained=False, freeze_backbone=False, in_chans=k,
                                          backbone=(teacher_backbone or backbone)).to(device)
        teacher.load_state_dict(torch.load(teacher_ckpt, map_location='cpu'), strict=True)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"[KD] teacher={teacher_ckpt} ({teacher_backbone or backbone}) frozen | "
              f"student=ImageNet-init {backbone} | alpha={kd_alpha} T={kd_temp}")

    # [부분 미세조정] train_stages 지정 시 해당 Swin stage(layers.i)만 학습, 나머지 backbone 동결.
    # 예: train_stages=[2,3] → 깊은/거친-스케일 stage 3·4 + head 만 학습(저수준 texture 고정).
    if train_stages is not None:
        for p in model.parameters():
            p.requires_grad = False
        keep = tuple(f'backbone.layers.{i}' for i in train_stages)
        train_stem = 0 in train_stages   # 저수준 stage 학습 시 patch_embed(stem)도 함께 해제
        for n, p in model.named_parameters():
            if n.startswith('head') or n.startswith('saliency_proj') \
               or n.startswith('backbone.norm') or any(k in n for k in keep) \
               or (train_stem and n.startswith('backbone.patch_embed')):
                p.requires_grad = True
        tn = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"[FreezeMode] training Swin stages {train_stages} + head/norm only: {tn:.1f}M trainable")

    # LLRD: stage 별 개별 param-group(층별 lr 감쇠). 아니면 학습 파라미터 단일 그룹(flat lr).
    if llrd:
        param_groups = build_llrd_groups(model, base_lr=lr, decay=llrd_decay,
                                         weight_decay=1e-2, lrs=llrd_lrs)
        optimizer = torch.optim.AdamW(param_groups)
        print(f"[LLRD] enabled: {len(param_groups)} groups, base_lr={lr:.2e}, "
              f"{'imbalanced lrs=' + str(llrd_lrs) if llrd_lrs else 'decay=' + str(llrd_decay)}")
    else:
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp_dtype = torch.bfloat16

    # weighted-CE: pos_weight = #neg/#pos over the train set (class-imbalance weighting).
    pos_weight_t = None
    if loss_type == 'wce':
        ys = np.array([int(r['label']) for r in train_ds.records])
        npos = max(1, int(ys.sum())); nneg = int(len(ys) - ys.sum())
        pos_weight_t = torch.tensor(float(nneg / npos), device=device)
        print(f"[loss] weighted-CE pos_weight={nneg}/{npos}={nneg/npos:.2f}")

    ckpt_dir = f'transfer_learning/{name}/checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)
    mode_tag = 'kd' if teacher_ckpt is not None else 'wt'
    ckpt_path = os.path.join(ckpt_dir, f'best_transfer_{mode_tag}_swin{arch_tag}_r{neighbor_slices}.pth')
    best_auroc, no_improve = 0.0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"[{name}] Epoch {epoch}/{epochs} [Train]", disable=None)
        for batch in pbar:
            volumes = batch['volumes'].to(device, non_blocking=True)
            depth_mask = batch['depth_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast('cuda', dtype=amp_dtype):
                logits, _ = model(volumes, depth_mask)   # [B, Z=1]
                scan_logits, _ = logits.max(dim=1)         # [B]
                if loss_type == 'wce':
                    loss = F.binary_cross_entropy_with_logits(
                        scan_logits, labels, pos_weight=pos_weight_t)
                else:
                    loss = focal_loss_with_logits(scan_logits, labels)
                if kd_mode:
                    with torch.no_grad():
                        t_logits, _ = teacher(volumes, depth_mask)
                        t_scan, _ = t_logits.max(dim=1)
                    kd = kd_bce_with_logits(scan_logits, t_scan, T=kd_temp)
                    loss = kd_alpha * kd + (1.0 - kd_alpha) * loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            pbar.set_postfix({'loss': f"{loss.item():.3f}"})
        scheduler.step()

        # ---- validation (model selection ONLY — TEST is never used for tuning) ----
        auroc, f1, acc, _, _ = _eval_split(model, val_loader, device, amp_dtype,
                                           desc=f"[{name}] Epoch {epoch}/{epochs} [Val]")
        print(f"--> [{name}] VAL AUROC: {auroc:.4f} | F1: {f1:.4f} | Acc: {acc:.4f}")

        if auroc > best_auroc:
            best_auroc, no_improve = auroc, 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"[*] Best saved → {ckpt_path} (AUROC {best_auroc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[EarlyStopping] stop at epoch {epoch} (best {best_auroc:.4f})")
                break
    print(f"[{name}] transfer done. best VAL AUROC={best_auroc:.4f}")

    # ---- held-out TEST (touched ONCE, on the best-val checkpoint) ----
    if eval_test:
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        model.to(device)
        test_ds = dataset_cls(split='test', neighbor_slices=neighbor_slices,
                              augment=False, cache_dir=cache_dir)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, collate_fn=collate_mammo2d,
                                 pin_memory=True)
        t_auroc, t_f1, t_acc, t_preds, t_labels = _eval_split(
            model, test_loader, device, amp_dtype, desc=f"[{name}] [TEST]")
        print(f"==> [{name}] TEST AUROC: {t_auroc:.4f} | F1: {t_f1:.4f} | Acc: {t_acc:.4f}")
        if preds_out:
            import json
            os.makedirs(os.path.dirname(preds_out), exist_ok=True)
            with open(preds_out, 'w') as f:
                json.dump({'name': name, 'arch_tag': arch_tag, 'seed': seed,
                           'val_auroc': best_auroc, 'test_auroc': t_auroc,
                           'test_f1': t_f1, 'test_acc': t_acc,
                           'preds': t_preds, 'labels': t_labels}, f)
            print(f"[preds] saved test predictions → {preds_out}")
        return {'val_auroc': best_auroc, 'test_auroc': t_auroc}
    return best_auroc
