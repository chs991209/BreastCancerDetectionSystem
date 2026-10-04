"""BUSI 전이학습 엔트리 (⚠️ 초음파 모달리티). PYTHONPATH=/workspace python transfer_learning/busi/run.py"""
from transfer_learning.busi.dataset import BUSIDataset
from transfer_learning.transfer_common import run_transfer

if __name__ == "__main__":
    run_transfer(
        BUSIDataset, name='busi', neighbor_slices=1,
        dbt_ckpt='checkpoints_b16/combined_25d_swinb_r1_pos10_strat_pureN_b16.pth',
        backbone='swin_base_patch4_window12_384', arch_tag='cb',
        train_stages=[0, 1],   # partial: Swin stages 3&4 only
        balance=True,    # normal 133 : abnormal 647 → 1:1 다운샘플
        epochs=15, batch_size=16, lr=1e-4,
    )
