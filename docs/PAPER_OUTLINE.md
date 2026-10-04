# Paper outline — DBT→2D transfer-learning for reproducible breast AI

**Target:** npj Digital Medicine. **Convention followed:** medical-AI norm — IMRaD + strong
**Discussion / Limitations / Future Outlook**, modest **Conclusion**. We characterize + validate some +
propose the rest; we do **not** claim to solve reproducibility (see `PROPOSED_METHODS_small_data.md`).

---

## Abstract (structured)
Background (repro crisis = lack of generalization; no breast foundation model; ID-only reporting) ·
Methods (combined DBT pretrain → 2D transfer; leakage-safe; OOD eval) · Results (DBT-TL > ImageNet-TL,
label-efficient; honest whole-image AUC; cross-region gap) · Conclusion (better *adaptable* foundation;
open problems remain).

## 1. Introduction
- Reproducibility crisis in breast AI = **lack of generalization**: no foundation model; large-data models
  stay dataset-specific; field reports **in-distribution** (train=test dataset, no external) → ~0.99 that
  doesn't reproduce. (cite Haibe-Kains 2020; "shortcuts" arXiv 2303.16417; MammoClean; geographic-equity.)
- **Clinical reality:** clinics adapt a model to *their own* data; they need an adaptable foundation.
- **Contribution:** (i) DBT (BCS-DBT + DBT-2026) pretraining as a better adaptable foundation than ImageNet;
  (ii) honest, leakage-safe, OOD/cross-region characterization; (iii) proposed remedies for the rest.

## 2. Methods
- Data: combined DBT (lesion, 3-class hierarchy normal/benign/malignant); 2D sets (EMBED·MIAS·CDD-CESM
  lesion; CMMD·RSNA·CBIS malignancy; INbreast·BUSI caveated). Label spec → `LABEL_DEFINITIONS.md`.
- Model: 2.5D-bridged Swin-B + MIL; transfer variants (ImageNet-TL / DBT-TL full·frozen·LP-FT).
- Splits: patient-grouped, leakage-safe; balancing configs (balanced vs natural prevalence).
- **Objectivity protocol:** ≥3 seeds, 95% CI, DeLong (Holm–Bonferroni), calibration (ECE), TEST-once,
  TRIPOD+AI/CLAIM; code + weights + split manifests released.

## 3. Results (tables T1–T4, `EXPERIMENT_AUDIT.md`)
- **T1** clinical-adaptation: DBT-TL vs ImageNet-TL per dataset (mean±CI, DeLong).
- **E8** label-efficiency curve (10/25/50/100% local data) — the clinical figure.
- **T2** balancing (balanced vs natural prevalence).
- **T3/E4** cross-region OOD gap; **E7** zero-shot cross-dataset (foundation probe).
- Level-2 malignancy (secondary). Literature-comparison table with "published setting" column.

## 4. Discussion
- **The 0.76-vs-0.99 gap is evaluation protocol (ID/patch/leaky vs OOD/whole-image/honest), NOT our
  pretraining** — under their protocol our model would also hit ~0.99 (`LITERATURE_AUC_comparison.md`).
- DBT pretraining helps *adaptation* (m1>m0), especially at low local-data (E8).
- Mini-dataset transfer is unstable (small n → high-variance AUC); honest CIs expose it.
- Regional gap (E4) shows no fixed model generalizes → motivates per-clinic adaptation.

## 5. Limitations (state openly)
Caveated datasets (INbreast/BUSI file-level leakage; kept, not headline); EMBED = single-institution (though
racially diverse); no breast foundation model yet; some datasets malignancy-only (no normal); modest absolute
AUC is the honest whole-image number, not leaderboard.

## 6. Future outlook (proposed, not solved — `PROPOSED_METHODS_small_data.md §B2`)
Breast foundation model via multi-source pretraining; domain adaptation (CORAL/DANN); test-time adaptation;
acquisition harmonization; few-shot/meta-learning for tiny clinic sets; uncertainty/abstention for safe
small-data deployment.

## 7. Conclusion (modest)
DBT pretraining is a **better adaptable foundation** for per-clinic breast-AI than ImageNet, delivered with
reproducible, leakage-safe, OOD-aware evaluation. It does not solve breast-AI generalization — but it
characterizes the gap honestly and offers concrete directions. Open problems remain, and we release
everything to reproduce.

## Back matter
Data/code availability (open datasets + released weights + split manifests + seeds); reporting checklists;
compute/env manifest.
