"""
[교차차원 지식증류 드라이버] 3D-DBT teacher → 2D student KD.
teacher 2종(Swin-B 0.7392, Swin-S 0.7822) × 2D dataset 3종 = 6 런.
student 는 각 teacher 와 동일 arch, ImageNet init(구조 지식은 KD loss 로만 전달).
실행: PYTHONPATH=/workspace python transfer_learning/run_kd.py
"""
from transfer_learning.transfer_common import run_transfer
from transfer_learning.vindr.dataset import VinDrMammoDataset
from transfer_learning.cmmd.dataset import CMMDDataset
from transfer_learning.inbreast.dataset import INbreastDataset

TEACHERS = [
    ('b', 'swin_base_patch4_window12_384', 'checkpoints/best_25d_r1.pth'),          # 0.7392
    ('s', 'swin_small_patch4_window7_224', 'checkpoints/best_25d_swins_r1_pos05.pth'),  # 0.7822
]
DATASETS = [
    (CMMDDataset,       'cmmd',     None, 10),
    (INbreastDataset,   'inbreast', None, 15),
    (VinDrMammoDataset, 'vindr',    0.20, 10),   # 5% 양성 → 20:80 균형
]

if __name__ == "__main__":
    results = {}
    for arch_tag, backbone, tckpt in TEACHERS:
        for cls, name, pos_rate, epochs in DATASETS:
            print(f"\n===== KD: teacher swin-{arch_tag} → {name} =====", flush=True)
            best = run_transfer(
                cls, name=name, neighbor_slices=1,
                backbone=backbone, arch_tag=arch_tag,          # student = same arch as teacher
                teacher_ckpt=tckpt, teacher_backbone=backbone, # frozen 3D teacher
                kd_alpha=0.5, kd_temp=2.0,
                pos_rate=pos_rate, epochs=epochs, batch_size=16, lr=1e-4,
            )
            results[f"{name}_swin{arch_tag}"] = best
    print("\n===== KD SUMMARY (best val AUROC) =====")
    for key, auroc in results.items():
        print(f"{key}: {auroc:.4f}")
