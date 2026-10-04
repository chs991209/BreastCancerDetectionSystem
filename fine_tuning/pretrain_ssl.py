"""
Phase B — SIFT-DBT 스타일 자기지도(Self-Supervised) 사전학습.

라벨 없이 ~19k train 볼륨의 모든 슬라이스로 Swin-B 백본을 MoCo v2 대조학습한다.
DBT 기하 구조를 활용해 양성쌍(positive pair)을 구성:
  - p=cross_view_p 확률로 '같은 study, 다른 view' 볼륨의 슬라이스
  - 아니면 같은 볼륨의 인접 슬라이스(±k)
음성(negative)은 메모리 큐(K개)로 공급. 학습된 백본만 저장하여
fine-tune.py 의 DimensionAgnostic3DSwin.backbone 에 그대로 로드한다.

전제: /data/cache/preproc/train/<PID>/<StudyUID>_<view>.npz (프리워밍 완료) 재사용 → 재디코딩 0.
실행(반드시 Phase A 종료 후, GPU 단독):
    python pretrain_ssl.py --epochs 100 --batch_size 256
"""
import os
import json
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.amp as amp
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import timm


# ==============================================================================
# 1. Dataset — 캐시된 .npz 슬라이스 위에서 라벨 없이 양성쌍 생성
# ==============================================================================
class DBTSliceSSL(Dataset):
    def __init__(self, index_json: str, preproc_dir: str,
                 img_size: int = 384, k_neighbor: int = 9, cross_view_p: float = 0.25,
                 repeat: int = 4):
        with open(index_json) as f:
            entries = json.load(f)

        # 캐시 경로 복원 + 존재하는 것만 채택
        self.items = []
        for e in entries:
            cp = os.path.join(preproc_dir, e['patient_id'], f"{e['study_uid']}_{e['view']}.npz")
            if os.path.exists(cp):
                self.items.append({'cache': cp, 'study': e['study_uid']})

        # cross-view 양성쌍을 위한 study -> [item index] 매핑
        self.study2idx = {}
        for i, it in enumerate(self.items):
            self.study2idx.setdefault(it['study'], []).append(i)

        self.k = k_neighbor
        self.cross_view_p = cross_view_p
        self.repeat = repeat   # epoch당 볼륨당 여러 번 샘플 → 더 많은 step/momentum 갱신
        # [증강은 mild] 미세석회화/미세종괴 보존을 위해 강한 블러·색공간 왜곡 배제.
        self.aug = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.8),
        ])
        print(f"[SSL-Dataset] {len(self.items)} volumes across {len(self.study2idx)} studies")

    def __len__(self):
        return len(self.items) * self.repeat

    @staticmethod
    def _load_vol(path):
        with np.load(path) as d:
            return d['img']  # uint8 [Z, H, W]

    def _to_tensor(self, slice_u8):
        return torch.from_numpy(slice_u8).float().unsqueeze(0) / 255.0  # [1,H,W]

    def __getitem__(self, i):
        it = self.items[i % len(self.items)]            # repeat 인덱싱
        vol = self._load_vol(it['cache'])               # [Z,H,W]
        z = np.random.randint(vol.shape[0])
        anchor = vol[z]

        siblings = [s for s in self.study2idx[it['study']] if s != i]
        if siblings and np.random.rand() < self.cross_view_p:
            # 같은 study, 다른 view 볼륨의 임의 슬라이스 (DBT 교차뷰 양성쌍)
            vol2 = self._load_vol(self.items[np.random.choice(siblings)]['cache'])
            key_slice = vol2[np.random.randint(vol2.shape[0])]
        else:
            # 같은 볼륨의 인접 슬라이스 (±k). offset 0 이면 표준 MoCo 증강쌍과 동일.
            z2 = int(np.clip(z + np.random.randint(-self.k, self.k + 1), 0, vol.shape[0] - 1))
            key_slice = vol[z2]

        # 두 view 각각 독립 증강
        view_q = self.aug(self._to_tensor(anchor))
        view_k = self.aug(self._to_tensor(key_slice))
        return view_q, view_k


# ==============================================================================
# 2. MoCo v2 — Swin-B 인코더 + projection MLP + 메모리 큐
# ==============================================================================
class Encoder(nn.Module):
    """Swin-B 백본(흑백) + 2-layer projection MLP. 저장 시 .backbone 만 추출."""
    def __init__(self, proj_dim: int = 128, pretrained: bool = False):
        super().__init__()
        self.backbone = timm.create_model(
            'swin_base_patch4_window12_384', pretrained=pretrained,
            in_chans=1, num_classes=0)            # → pooled feature [N, C]
        feat = self.backbone.num_features
        self.proj = nn.Sequential(
            nn.Linear(feat, feat), nn.ReLU(inplace=True), nn.Linear(feat, proj_dim))

    def forward(self, x):
        return self.proj(self.backbone(x))        # [N, proj_dim]


class MoCo(nn.Module):
    def __init__(self, proj_dim=128, K=4096, m=0.999, T=0.2, pretrained=False,
                 grad_checkpoint=True):
        super().__init__()
        self.m, self.T, self.K = m, T, K
        self.enc_q = Encoder(proj_dim, pretrained=pretrained)
        # [VRAM] query 인코더 백본에 gradient checkpointing → batch_size 256@384 도 수용.
        # (MoCo 는 메모리 큐가 음성을 공급하므로 거대한 배치가 필수는 아님 → 작게도 무방.)
        if grad_checkpoint:
            try:
                self.enc_q.backbone.set_grad_checkpointing(True)
            except Exception:
                pass
        self.enc_k = copy.deepcopy(self.enc_q)
        for p in self.enc_k.parameters():
            p.requires_grad = False               # key encoder 는 momentum 으로만 갱신

        self.register_buffer('queue', nn.functional.normalize(torch.randn(proj_dim, K), dim=0))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update(self):
        for pq, pk in zip(self.enc_q.parameters(), self.enc_k.parameters()):
            pk.data = pk.data * self.m + pq.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_enqueue(self, keys):
        keys = keys.float()
        bs = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + bs <= self.K:
            self.queue[:, ptr:ptr + bs] = keys.T
        else:                                     # 큐 끝 wrap-around
            first = self.K - ptr
            self.queue[:, ptr:] = keys[:first].T
            self.queue[:, :bs - first] = keys[first:].T
        self.queue_ptr[0] = (ptr + bs) % self.K

    def forward(self, im_q, im_k):
        q = nn.functional.normalize(self.enc_q(im_q), dim=1)        # [N, d]
        with torch.no_grad():
            self._momentum_update()
            k = nn.functional.normalize(self.enc_k(im_k), dim=1)    # [N, d]

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)      # [N,1]
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])  # [N,K]
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T         # 양성=index 0
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss = nn.functional.cross_entropy(logits, labels)
        acc = (logits.argmax(1) == labels).float().mean()          # 대조 정확도(top-1)
        self._dequeue_enqueue(k)
        return loss, acc


# ==============================================================================
# 3. Train
# ==============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index_json', default='/data/cache/bcsdbt_train_index_v2.json')
    ap.add_argument('--preproc_dir', default='/data/cache/preproc/train')
    ap.add_argument('--out', default='checkpoints/ssl_swinb_backbone.pth')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=192)   # 256→192: VRAM 여유(46GB 근접 회피)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=0.05)
    ap.add_argument('--num_workers', type=int, default=12)
    ap.add_argument('--K', type=int, default=4096)
    ap.add_argument('--m', type=float, default=0.99)         # 0.999→0.99: step 적은 환경서 key encoder 추종 가속
    ap.add_argument('--T', type=float, default=0.2)
    ap.add_argument('--proj_dim', type=int, default=128)
    ap.add_argument('--cross_view_p', type=float, default=0.25)  # 0.5→0.25: 어려운 교차뷰 비중↓
    ap.add_argument('--repeat', type=int, default=4)         # epoch당 step↑ (momentum 갱신↑)
    ap.add_argument('--scratch', action='store_true',
                    help='지정 시 ImageNet 미사용(무작위 초기화). 기본=ImageNet 웜스타트')
    ap.add_argument('--save_every', type=int, default=10)
    ap.add_argument('--no_grad_checkpoint', action='store_true',
                    help='query 인코더 gradient checkpointing 비활성(소배치에서만 권장)')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    ds = DBTSliceSSL(args.index_json, args.preproc_dir, img_size=384,
                     cross_view_p=args.cross_view_p, repeat=args.repeat)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=(args.num_workers > 0),
                        prefetch_factor=4 if args.num_workers > 0 else None)

    moco = MoCo(proj_dim=args.proj_dim, K=args.K, m=args.m, T=args.T,
                pretrained=not args.scratch,                 # 기본 ImageNet 웜스타트
                grad_checkpoint=not args.no_grad_checkpoint).to(device)
    print(f"[SSL] init={'scratch' if args.scratch else 'ImageNet'} m={args.m} "
          f"cross_view_p={args.cross_view_p} repeat={args.repeat} bs={args.batch_size}")
    opt = torch.optim.AdamW(moco.enc_q.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    print(f"[SSL] device={device} batches/epoch={len(loader)} batch_size={args.batch_size}")

    for epoch in range(1, args.epochs + 1):
        moco.train()
        run_loss = run_acc = 0.0
        pbar = tqdm(loader, desc=f"SSL Epoch {epoch}/{args.epochs}")
        for im_q, im_k in pbar:
            im_q = im_q.to(device, non_blocking=True)
            im_k = im_k.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with amp.autocast('cuda', dtype=torch.bfloat16):
                loss, acc = moco(im_q, im_k)
            loss.backward()
            opt.step()
            run_loss += loss.item(); run_acc += acc.item()
            pbar.set_postfix({'loss': f"{loss.item():.3f}", 'top1': f"{acc.item():.3f}"})
        sched.step()
        n = len(loader)
        print(f"--> [SSL] Epoch {epoch} loss={run_loss/n:.4f} top1={run_acc/n:.4f}", flush=True)

        if epoch % args.save_every == 0 or epoch == args.epochs:
            # ONLY the Swin-B backbone → DimensionAgnostic3DSwin.backbone 에 그대로 로드
            torch.save(moco.enc_q.backbone.state_dict(), args.out)
            print(f"[SSL] backbone saved → {args.out} (epoch {epoch})", flush=True)

    print("[SSL] Phase B complete. Load into fine-tune.py via model.backbone.load_state_dict(...).")


if __name__ == "__main__":
    main()
