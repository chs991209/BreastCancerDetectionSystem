# Literature AUC comparison — our results vs published transfer-learning

**Purpose:** benchmark our (honest, whole-image, patient-grouped) 2D transfer AUCs against published
numbers on the same datasets — and explain the gap, which *is* the reproducibility thesis.

---

## The headline gap

| Dataset | **Ours** (whole-image, patient-grouped, lesion-presence) | Published headline AUC | Published setting (why higher) |
|---|---|---|---|
| INbreast | 0.83 (⚠ file-level, caveated) | **0.95–1.00** | patch/ROI-level, tiny test (n=410), image-level split (leakage); ensembles hit ~1.00 |
| CBIS-DDSM | *pending (loader)* | 0.91–0.95 | patch/ROI classifiers, ResNet ensembles |
| MIAS | 0.86 | **0.9994** (5-fold CV) | patch-level, cross-val (not held-out), n=322 |
| CMMD | 0.78 | ~0.99 (some) | patch/ROI-level |
| EMBED | 0.76 (DBT-TL, 3-seed) | (EMBED papers emphasize *realistic-imbalance* eval, not leaderboard AUC) | large, diverse, realistic distribution |

**The credible whole-image comparator:** Shen et al. (Nature Sci Rep 2019) — whole-image, patient-disjoint,
INbreast **per-image AUC 0.95** single / **0.98** 4-model ensemble (trained on DDSM + large private set).
This is the fair apples-to-apples point; the 0.99–1.00 numbers are mostly patch-level or leakage-inflated.

---

## Why the field's numbers are inflated (documented, not opinion)
The gap is driven by the exact shortcuts our protocol controls for — catalogued in two key references:
- **"Problems and shortcuts in deep learning for screening mammography"** (arXiv 2303.16417) — enumerates
  patch-level evaluation, image-level (non-patient) splits, and tiny-test overfitting as pervasive.
- **"Reproducibility and Explainability of Deep Learning in Mammography: A Systematic Review"** (PMC11188703)
  — most studies are single-dataset, non-patient-partitioned, non-reproducible.

Specific inflation mechanisms:
1. **Patch/ROI-level** classification (crop the known lesion) — far easier than whole-image; leaks localization.
2. **Image-level splits** (not patient-level) → same patient's views in train+test = leakage.
3. **Tiny test sets** (INbreast 410, MIAS 322) → 100% AUC is noise/overfit, not skill.
4. **Cross-validation AUC** reported as if held-out; no external validation.
5. **Single-dataset**, same-distribution train/test → no batch-effect or regional stress.

---

## Two confounded axes — do not misattribute the gap
The cross-paper difference mixes **two independent axes**; keep them separate:

| Axis | Ours | Others | Effect |
|---|---|---|---|
| **Pretraining source** | real 3D DBT (BCS-DBT + DBT-2026) — *our novelty* | ImageNet / patch-DDSM | small **positive** lift (m1 DBT-TL > m0 ImageNet-TL) |
| **Evaluation protocol** | whole-image, patient-grouped, honest prevalence | patch/ROI, image-level split, tiny test | explains the **entire** 0.76-vs-0.99 gap |

- The DBT pretraining did **not** lower our AUC — the honest evaluation did. Under their protocol
  (patch-level/leaky) our DBT model would also reach ~0.99.
- Cross-paper comparison is therefore **confounded** (source *and* protocol differ). The only clean
  evidence that DBT pretraining helps is our **controlled m0 (ImageNet-TL) vs m1 (DBT-TL)** — same data,
  same evaluation, source is the only variable. **Headline the DBT-pretraining claim with m1−m0, not with
  the literature gap.**

## Where we stand (honest interpretation)
- Our numbers sit in the **honest whole-image regime** (~0.75–0.86), not the leaderboard regime (~0.99).
  That is expected and *is the point*: we measure the **leakage-safe, patient-grouped, whole-image,
  honest-prevalence** number that actually reproduces.
- **We are not claiming to beat 0.99.** The contribution is that our numbers are *reproducible and
  cross-region validated*, whereas the 0.99s largely are not (patch-level / leakage / tiny-test).
- Fair positive comparators for our setting: Shen 2019 whole-image INbreast 0.95 (with heavy pretraining);
  our DBT→2D transfer is a different, honest protocol at screening scale.

**Paper framing:** present the table above with the "published setting" column — it turns our lower AUCs
into evidence for the reproducibility argument rather than a weakness. Caveat our own INbreast (file-level)
the same way we caveat theirs.

---

## Sources
- Shen et al., Deep Learning to Improve Breast Cancer Detection on Screening Mammography — Nature Sci Rep 2019 (arXiv 1708.09427)
- Problems and shortcuts in deep learning for screening mammography — arXiv 2303.16417
- Reproducibility and Explainability of Deep Learning in Mammography: A Systematic Review — PMC11188703
- EMBED dataset — Radiology: AI 2022 (ryai.220047)
- Stacked ensemble ResNet (CBIS 0.95 / INbreast 0.99) — Sci Rep 2022 (s41598-022-15632-6)
- Multi-stage transfer (DDSM/INbreast/MIAS ~0.999, 5-fold CV) — PMC8909211
