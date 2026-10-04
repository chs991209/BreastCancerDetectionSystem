"""Stage-1 DBT fine-tune from SCRATCH (random init). Ablation vs the ImageNet-init checkpoint.
Controlled: ALL hyperparameters identical to the ImageNet-DBT run; only pretrained=False differs.
실행: PYTHONPATH=/workspace python fine_tuning_combined/train_scratch.py
"""
import sys
sys.path.insert(0, '/workspace')
import importlib.util
spec = importlib.util.spec_from_file_location("ftc", "/workspace/fine_tuning_combined/fine-tune_combined.py")
ftc = importlib.util.module_from_spec(spec); spec.loader.exec_module(ftc)

if __name__ == "__main__":
    # identical to the ImageNet-DBT run (epochs/batch/pos_rate/LLRD/warmup defaults); only init differs.
    ftc.train_combined(epochs=15, batch_size=16, pos_rate=0.10, neighbor_slices=1,
                       num_workers=8,
                       backbone='swin_base_patch4_window12_384', arch_tag='bscratch',
                       ckpt_dir='checkpoints_b16', pretrained=False)
