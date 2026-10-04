"""
[Natural EMBED 전용 러너] 실행 중인 메인 매트릭스(구 코드 로드 → embed_nat 미포함)와 충돌 없이
embed_nat 조합만 실행. preds 명명은 메인과 동일 → summarize() 가 그대로 취합.
methods × {balanced, plain} × seeds. TEST 는 항상 자연 유병률(36.5%).
실행: PYTHONPATH=/workspace python experiments/embed_nat_matrix.py
"""
from experiments.multiseed_matrix import run_one, SEEDS, METHODS, DATASETS

if __name__ == "__main__":
    ds_cls, task, bals = DATASETS['embed_nat']
    combos = [(m, b) for m in METHODS for b in bals]
    total = len(combos) * len(SEEDS)
    i = 0
    for seed in SEEDS:
        for method, balance in combos:
            i += 1
            bt = 'bal' if balance else 'plain'
            print(f"\n##### [embed_nat {i}/{total}] {method}/{bt}/s{seed} #####", flush=True)
            try:
                au = run_one('embed_nat', ds_cls, method, balance, seed)
                print(f"##### RESULT embed_nat/{method}/{bt}/s{seed}: TEST AUROC={au:.4f}")
            except Exception as e:
                print(f"##### FAILED embed_nat/{method}/{bt}/s{seed}: {type(e).__name__}: {e}")
    print("\n===== embed_nat COMPLETE =====")
