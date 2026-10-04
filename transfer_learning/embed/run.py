"""EMBED 전이학습 엔트리. 실행: PYTHONPATH=/workspace python transfer_learning/embed/run.py"""
from transfer_learning.embed.dataset import EMBEDDataset
from transfer_learning.transfer_common import run_transfer

if __name__ == "__main__":
    run_transfer(
        EMBEDDataset, name='embed', neighbor_slices=1,
        # 소스 = 통합 lesion Swin-B batch16 (held-out TEST 0.929).
        dbt_ckpt='checkpoints_b16/combined_25d_swinb_r1_pos10_strat_pureN_b16.pth',
        backbone='swin_base_patch4_window12_384', arch_tag='cb',
        balance=True,    # 대칭 1:1 무작위 다운샘플
        train_stages=[2, 3],   # Swin stage 3·4(깊은/거친-스케일)만 학습, 저수준 stage 동결
        epochs=10, batch_size=16, lr=1e-4,
    )
