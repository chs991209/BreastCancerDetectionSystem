"""MIAS 전이학습 엔트리. 실행: PYTHONPATH=/workspace python transfer_learning/mias/run.py"""
from transfer_learning.mias.dataset import MIASDataset
from transfer_learning.transfer_common import run_transfer

if __name__ == "__main__":
    run_transfer(
        MIASDataset, name='mias', neighbor_slices=1,
        # 소스 = 통합 lesion Swin-B (pureN). MIAS 도 lesion-presence 라 과제 정합성 높음.
        dbt_ckpt='checkpoints_b16/combined_25d_swinb_r1_pos10_strat_pureN_b16.pth',
        backbone='swin_base_patch4_window12_384', arch_tag='cb',
        train_stages=[0, 1],   # partial: Swin stages 3&4 only
        balance=True,    # 대칭 1:1 무작위 다운샘플
        epochs=15, batch_size=16, lr=1e-4,   # 322장 소규모 → epochs 여유
    )
