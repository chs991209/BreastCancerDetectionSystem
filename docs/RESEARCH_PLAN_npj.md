# Research Plan & Continual-Work Tracker — npj Digital Medicine

**Target journal:** npj Digital Medicine (Nature Portfolio) · **Paper thesis:** reproducibility of medical-AI systems, demonstrated on a DBT→2D breast-lesion pipeline.
**Living document** — update status columns as work completes. Last updated 2026-08-25.

> **Why this doc exists.** npj Digital Medicine reviewers expect (1) statistical rigor — multi-seed,
> confidence intervals, significance tests; (2) data-integrity guarantees — no leakage, honest prevalence;
> (3) calibration and generalization evidence, not AUROC alone; (4) checklist-compliant reporting
> (TRIPOD+AI / CLAIM). Weak rigor and hidden leakage are the two most common reasons for revision or
> rejection. Every continual experiment below is designed to be **objective and pre-specified** so results
> cannot be dismissed as cherry-picked.

## Scope of contribution (ambition — read first)
We do **not** claim to solve reproducibility. The paper (1) **characterizes** the generalization/repro
problems rigorously, (2) **validates** DBT pretraining as a better *adaptable* foundation for per-clinic
fine-tuning (leakage-safe, honest, label-efficient), and (3) **proposes methods** for the residual
mini-dataset / OOD problems — some validated, some offered as directions we cannot fully solve now.
Honest disclosure of what remains open is itself part of the reproducibility contribution.
See `PROPOSED_METHODS_small_data.md` for the problem→proposed-method map.

## 0. Definition of the reproducibility problem we target (thesis)
**Lack of reproducibility = lack of generalization.** Concretely:
1. **No breast foundation model** — no parameter set generalizes across the breast domain.
2. Even large-data training (BCS-DBT + DBT-2026) yields **dataset-specific** parameters that do not
   transfer to *most* breast data.
3. The field reports **in-distribution (ID)** results — train and test on the **same** dataset, **no
   external data** — so headline AUCs (~0.99) do **not** reproduce on unseen data.

**Therefore the honest measure is out-of-distribution (OOD) / external evaluation**: test on a
dataset/region the model was **not** trained or fine-tuned on. This is what the field avoids by staying ID,
and it is the number our paper reports.

**Two distinct, both-valid claims (do not conflate):**

| Claim | Question | Experiment | Role |
|---|---|---|---|
| **Clinical adaptation (the product)** | Given our pretrained model, can a clinic fine-tune it to **their own dataset** and get a better, more reproducible local model than the ImageNet start? | **m1 (DBT-TL) vs m0 (ImageNet-TL)**, per-dataset fine-tune + **label-efficiency curve (E8)** | **primary / product** |
| **Generalization (the *why*)** | Does any single fixed model generalize across datasets without adaptation? (No — that's the reproducibility crisis.) | **cross-region E4 · zero-shot cross-dataset E7** | **motivates** adaptation |

- **Clinical reality:** no site deploys a foreign model as-is; they adapt it to local data. So the
  **in-domain fine-tuned m1 is the real use case**, not a weakness. Our contribution = DBT pretraining
  is a **better adaptable foundation** than ImageNet (m1 > m0), ideally with **fewer local labels** (E8).
- **The OOD results (E4/E7) explain *why* adaptation is required** — no model generalizes zero-shot —
  so the reproducibility problem *motivates* the per-clinic-adaptation solution rather than undermining it.
- m1 answers the clinical claim; it is *not* an OOD/generalization result. Keep the two tables separate.

---

## 1. Objectivity protocol — FREEZE before running headline experiments

These rules are fixed *before* the runs, not chosen after seeing results. Any deviation is logged.

| Rule | Specification |
|---|---|
| **Primary endpoint** | Held-out **TEST** AUROC on the pre-declared clean sets. Declared *before* runs. |
| **Multi-seed** | Every headline number = **N=5 seeds**; report **mean ± 95% CI** (across-seed + bootstrap on the test set). No single-run numbers in the main tables. |
| **Test set touched once** | All model/hyperparameter selection on **val only**. TEST is evaluated once per final model — never used for tuning or early-stopping. |
| **Significance** | AUROC comparisons via **DeLong test** (paired, same test set); report p-value + ΔAUROC. Correct for multiple comparisons (Holm–Bonferroni). |
| **Calibration** | Report **ECE + reliability diagram**; apply temperature scaling on val, report pre/post. |
| **No selective reporting** | Report **all** datasets and **all** ablation cells, including negative/null results. Failed ideas stay in the paper (or supplement). |
| **Determinism** | Global seed + DataLoader generator + worker seeding (done); fixed deterministic splits (done); **no synthetic per-epoch resampling** (done). |
| **Reporting standards** | Fill **TRIPOD+AI** and **CLAIM** checklists; release code + split manifests + model card; state compute + failures. |

---

## 2. Known weaknesses → anticipated reviewer critique → mitigation

Face these *before* a reviewer does. Status: ✅ done · 🔄 in progress · ⬜ to do.

| Weakness | Reviewer critique | Mitigation | Status |
|---|---|---|---|
| INbreast/BUSI **file-level split** (no patient IDs) | "Patient/view leakage inflates these numbers." | Recover patient IDs if possible; else **drop from headline**, keep only as disclosed caveated sanity checks. | ⬜ P0 |
| **Single-seed** point estimates | "No variance — could be a lucky seed." | Track A: 5-seed CIs on all headline numbers. | ⬜ P0 |
| INbreast **tiny val** (9 abnormal) | "Estimate is noise." | Drop from headline OR merge val+test with disclosed n; never headline a 48-sample set. | ⬜ P0 |
| **Label heterogeneity** across sets (biopsy vs BI-RADS; lesion vs malignancy) | "You pooled incomparable labels." | Never pool; per-dataset reporting + a **label-provenance table** (source, biopsy%, definition). | 🔄 P1 |
| EMBED **balanced 50:50 subset** ≠ screening prevalence | "Real screening is ~0.5% cancer; your AUROC is optimistic." | Add a **prevalence-realistic eval** (natural ratio) or explicitly scope claims to the balanced-subset setting. | ⬜ P1 |
| BUSI **ultrasound** modality mismatch | "Different modality — not a mammography result." | Frame BUSI as a **cross-modality sanity check**, excluded from the core DBT→FFDM claim. | 🔄 P1 |
| Undersampling for class balance | "Balancing changes the operating point." | Now removed (plain ratio, deterministic); report both balanced and natural where relevant. | ✅ |

---

## 3. Continual-work tracks (organized & prioritized)

**Track A — Statistical rigor** *(P0, blocks submission)*
- 5-seed reruns of the DBT combined model + each clean 2D transfer → mean ± 95% CI.
- DeLong tests for the key comparisons (combined vs single-dataset; full-FT vs LLRD vs partial).
- Bootstrap CIs on all reported TEST AUROCs.

**Track B — Transfer enhancement** *(P1; every variant evaluated under the §1 protocol)*
- LP-FT (linear-probe → fine-tune), 2.5D channel re-map, synthetic-2D (C-view) bridge, EMA. See `docs/3D_2D_전이_성능향상_방안.md`.
- **Rule:** a method enters the paper only if it beats baseline with non-overlapping 95% CI or a significant DeLong test — not on a single-run bump.

**Track C — Data integrity** *(P0)*
- Resolve INbreast/BUSI leakage (recover IDs or demote). Build the label-provenance table. Publish split manifests (patient-level) as a supplement.

**Track D — Calibration & prevalence realism** *(P1)*
- ECE + reliability diagrams; temperature scaling. Prevalence-realistic evaluation for at least one FFDM set.

**Track E — Reporting & reproducibility artifacts** *(P1)*
- TRIPOD+AI + CLAIM checklists; model card; code/data-availability statement; environment + seed manifest; compute + carbon note.

**Track F — External validation / generalization** *(P2)*
- Frame the multi-region 2D sets (China/UK/Portugal/USA) as a **region-shift generalization** analysis. Candidate additional external set: RSNA-BCD (biopsy + screening prevalence).

---

## 4. Near-term experiment queue (ordered, concrete)

| # | Experiment | Depends on | Status |
|---|---|---|---|
| 1 | EMBED plain-vs-imbalanced LLRD ablation | — | 🔄 running |
| 2 | Decide headline set (clean: EMBED/CMMD/MIAS) + disposition of INbreast/BUSI | §2 | ⬜ |
| 3 | 5-seed CIs: DBT combined model (VAL+TEST) | seeding ✅ | ⬜ |
| 4 | 5-seed CIs: each clean 2D transfer, best config | #2 | ⬜ |
| 5 | DeLong tests for headline comparisons | #3,#4 | ⬜ |
| 6 | LP-FT + channel re-map, multi-seed vs baseline | #4 | ⬜ |
| 7 | Calibration (ECE + temp scaling) on final models | #3,#4 | ⬜ |
| 8 | Label-provenance table + split-manifest supplement | — | ⬜ |

---

## 5. Decisions needed from the user

1. **Seeds:** N=5 recommended (3 minimum for CIs). Confirm N.
2. **Headline set:** propose EMBED · CMMD · MIAS (all patient-grouped, biopsy/pathology-grounded). Confirm.
3. **INbreast / BUSI:** demote to caveated supplement, or attempt patient-ID recovery for a clean re-split?
4. **Prevalence:** add a natural-prevalence eval, or scope claims to the balanced setting?

Once 1–4 are set, Track A (statistical rigor) starts — it is the submission-blocking prerequisite.
