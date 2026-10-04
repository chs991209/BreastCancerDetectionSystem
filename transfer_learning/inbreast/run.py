"""INbreast 전이학습 엔트리. 실행: PYTHONPATH=/workspace python transfer_learning/inbreast/run.py"""
from transfer_learning.inbreast.dataset import INbreastDataset
from transfer_learning.transfer_common import run_transfer

if __name__ == "__main__":
    run_transfer(
        INbreastDataset, name='inbreast', neighbor_slices=1,
        # DBT(3D 2.5D)→INbreast 가중치 전이(방향 B). 소스 = 통합 lesion Swin-B (held-out TEST 0.906).
        dbt_ckpt='checkpoints_b16/combined_25d_swinb_r1_pos10_strat_pureN_b16.pth',
        backbone='swin_base_patch4_window12_384', arch_tag='cb',
        train_stages=[0, 1],   # partial: Swin stages 3&4 only
        balance=True,    # 대칭 1:1 무작위 다운샘플 (normal 다수 축소)
        epochs=15, batch_size=16, lr=1e-4,  # 410장 소규모 → epochs 여유
    )
