"""VinDr-Mammo 전이학습 엔트리. 실행: PYTHONPATH=/workspace python transfer_learning/vindr/run.py"""
from transfer_learning.vindr.dataset import VinDrMammoDataset
from transfer_learning.transfer_common import run_transfer

if __name__ == "__main__":
    run_transfer(
        VinDrMammoDataset, name='vindr', neighbor_slices=1,
        # DBT(3D 2.5D)→VinDr 가중치 전이(방향 B). 소스 = 통합 lesion Swin-B (held-out TEST 0.906).
        dbt_ckpt='checkpoints/combined_25d_swinb_r1_pos10_strat_pureN.pth',
        backbone='swin_base_patch4_window12_384', arch_tag='cb',
        balance=True,    # 대칭 1:1 무작위 다운샘플 (normal 다수 축소)
        epochs=10, batch_size=16, lr=1e-4,
    )
