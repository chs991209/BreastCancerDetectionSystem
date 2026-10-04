"""EMBED 부분전이 3-way ablation (aligned split). full + freeze-deep (freeze-shallow 는 기존 0.760).
실행: PYTHONPATH=/workspace python transfer_learning/embed/ablation.py"""
from transfer_learning.embed.dataset import EMBEDDataset
from transfer_learning.transfer_common import run_transfer

CK = 'checkpoints_b16/combined_25d_swinb_r1_pos10_strat_pureN_b16.pth'
common = dict(neighbor_slices=1, dbt_ckpt=CK, backbone='swin_base_patch4_window12_384',
              balance=True, epochs=10, batch_size=16, lr=1e-4)

if __name__ == "__main__":
    print("\n##### EMBED full fine-tune #####", flush=True)
    full = run_transfer(EMBEDDataset, name='embed', arch_tag='cbfull', train_stages=None, **common)
    print("\n##### EMBED freeze-deep (train shallow) #####", flush=True)
    deep = run_transfer(EMBEDDataset, name='embed', arch_tag='cbshallow', train_stages=[0, 1], **common)
    print(f"\n=== EMBED ablation: full={full:.4f} | freeze-deep(train-shallow)={deep:.4f} ===")
