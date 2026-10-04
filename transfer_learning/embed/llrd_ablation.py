"""EMBED LLRD ablation — 재현성 중심(논문용).
공통: balance=False (합성 랜덤 언더샘플링 없음, plain ratio), seed=42 고정, full fine-tune(+LLRD).
- plain LLRD      : 기하 감쇠 lr = base * 0.75^depth (head→stem 단조 감소).
- imbalanced LLRD : depth별 배수 [head,stage4,stage3,stage2,stage1,stem].
    '병변의 미세한 차이' 가설 → 고해상 fine stage(stage1/2=depth 4,3)와 head 를 크게,
    거친 deep stage4(depth1)·generic stem(depth5)을 작게(비단조 bump).
실행: PYTHONPATH=/workspace python transfer_learning/embed/llrd_ablation.py
"""
from transfer_learning.embed.dataset import EMBEDDataset
from transfer_learning.transfer_common import run_transfer

CK = 'checkpoints_b16/combined_25d_swinb_r1_pos10_strat_pureN_b16.pth'
# depth: 0=head/norm 1=stage4 2=stage3 3=stage2 4=stage1 5=stem
IMBAL = [1.0, 0.35, 0.6, 1.0, 1.0, 0.25]

common = dict(neighbor_slices=1, dbt_ckpt=CK, backbone='swin_base_patch4_window12_384',
              balance=False, pos_rate=None,   # plain ratio — no synthetic random-sampling
              epochs=10, batch_size=16, lr=1e-4, train_stages=None,  # full fine-tune
              llrd=True, seed=42)

if __name__ == "__main__":
    print("\n##### EMBED  plain LLRD (geometric decay=0.75), plain ratio #####", flush=True)
    plain = run_transfer(EMBEDDataset, name='embed', arch_tag='llrdplain',
                         llrd_decay=0.75, **common)

    print("\n##### EMBED  imbalanced LLRD (fine-stage boost), plain ratio #####", flush=True)
    imbal = run_transfer(EMBEDDataset, name='embed', arch_tag='llrdimbal',
                         llrd_lrs=IMBAL, **common)

    print(f"\n=== EMBED LLRD ablation (plain ratio, seed=42) ===")
    print(f"    plain LLRD (decay 0.75)   : {plain:.4f}")
    print(f"    imbalanced LLRD {IMBAL}: {imbal:.4f}")
