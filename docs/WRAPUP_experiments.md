# Transfer-Learning Wrap-Up — experiments to finish the paper

**Scope:** transfer-learning only. Task hierarchy (see `LABEL_DEFINITIONS.md §1`):
**Level 1 = lesion detection** (normal vs benign+malignant, DEFAULT) · **Level 2 = malignancy**
(benign vs malignant, a division *within* the lesion process). Default method = weight-transfer + full-FT (m1).
All under the objectivity protocol (≥3 seeds, 95% CI, DeLong, TEST-once).

Status legend: ✅ done · 🔄 running · ➕ to add (this wrap-up).

---

## The finishing experiment set

| ID | Experiment | Datasets | Purpose (paper claim it supports) | Status |
|---|---|---|---|---|
| **E1** | **Level-1 lesion transfer** — methods m0–m3 × balancing {bal, plain} × 3 seeds | EMBED, EMBED-nat, MIAS, CDD-CESM | Core result: does DBT→2D transfer help lesion detection; which transfer method wins | 🔄 running (matrix) |
| **E2** | **Level-2 malignancy division** — benign vs malignant, m0–m3 × 3 seeds | CMMD, CDD-CESM, MIAS (EMBED opt.) | Completes the hierarchy: transfer for the malignant-vs-benign split | ➕ add |
| **E3** | **Balancing ablation** — balanced vs natural/plain prevalence | EMBED-nat + all imbalanced | Reproducibility: prevalence inflation (balanced ≠ honest) | 🔄 (E1) / ➕ (E2) |
| **E4** ★ | **Cross-region generalization (OOD)** — train region A → test region B, no shared data | EMBED(USA) · MIAS(UK) · CDD-CESM(Egypt) | **The reproducibility headline** — the OOD/external gap the field avoids (ID-only) | ➕ add |
| **E7** ★ | **Frozen / zero-shot cross-dataset** — DBT params → unseen dataset, no/min target fine-tune | all lesion sets | **Foundation-model / parameter-generalization probe** (do DBT params generalize?) | ➕ add |
| **E5** | **Calibration** — ECE + reliability, ± temperature scaling | final models | Reviewer requirement: AUROC alone is insufficient | ➕ add (post-hoc from saved preds) |
| **E6** | **Statistical finish** — mean±95%CI across seeds + DeLong (Holm–Bonferroni) | all cells | Statistical validity (no single-run claims) | ➕ add (post-hoc from preds) |

| **E8** ◆ | **Label/data-efficiency curve** — fine-tune on 10/25/50/100% of local train, DBT-TL vs ImageNet-TL | EMBED, MIAS, CDD-CESM | **The clinical-value figure** — reach good local performance with fewer local labels | ➕ add |

★ = reproducibility thesis (OOD/external): the field stays ID; we test cross-region (E4) / zero-shot (E7).
◆ = clinical-adaptation claim (the product): a clinic fine-tunes to **their own** data; DBT pretraining
beats the ImageNet start (E1 m1 vs m0) and, per E8, needs **fewer local labels**. The in-domain m1 is the
clinical use case here — *not* an OOD result. Two separate claims/tables — see `RESEARCH_PLAN_npj.md §0`.

RSNA stays as the **external-validation / realistic-prevalence anchor** (normal vs malignant-lesion, partial Level 1).

---

## Why this set wraps up the paper
Together they answer every question a transfer-learning + reproducibility paper must:
1. **Does the DBT→2D transfer work, and which method?** → E1 (m0 baseline vs m1 default vs m2/m3).
2. **Does it hold for the finer malignant/benign split?** → E2 (Level 2).
3. **Do the numbers reproduce at honest prevalence?** → E3 (balanced vs natural) + RSNA.
4. **Do they reproduce across regions/populations?** → E4 (the key novelty).
5. **Are the probabilities trustworthy, not just ranked?** → E5 (calibration).
6. **Are the claims statistically real, not lucky seeds?** → E6 (CI + DeLong).

After E1–E6, the transfer-learning story is complete and defensible; paper tables T1–T4 (`EXPERIMENT_AUDIT.md`) fill in.

---

## Execution order (single A6000, queued behind the running E1 matrix)
1. **E1** finishing (matrix, ~running).
2. **E2** malignancy matrix — queued behind E1.
3. **E4** cross-region — queued behind E2.
4. **E5 / E6** — post-hoc from `experiments/preds/*.json` (no GPU), run as preds accumulate.
