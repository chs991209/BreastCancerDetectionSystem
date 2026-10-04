# Label Definitions & Concepts — Combined DBT Pipeline

**Last updated:** 2026-08-10

**Authoritative** label reference for the pipeline (combined BCS-DBT + DBT-2026,
fine-tune → transfer-learning). These definitions are the main ones; earlier
malignant-only framing is retired (one-line note in §4).

---

## 1. Core concept — three classes, hierarchical decomposition

The vocabulary is **three classes: normal · benign · malignant**.

| Class | Meaning |
|---|---|
| **normal** | no lesion present |
| **benign** | a benign lesion present |
| **malignant** | a malignant lesion present |

**Key relation: `malignant ⊂ lesion`** — every malignant case is a lesion-involving case. So the
three classes decompose into a **two-level hierarchy**:

```
                 all cases
                /          \
           normal          lesion  (= benign ∪ malignant)   ← LEVEL 1: lesion detection (DEFAULT)
                          /        \
                    benign          malignant               ← LEVEL 2: malignancy (division within lesion)
```

- **Level 1 — lesion detection (the default / primary transfer task):** `normal` vs `lesion`,
  where **lesion = benign + malignant**. Ratio convention: always **normal : abnormal** (abnormal = lesion).
- **Level 2 — malignancy (a division *within* the lesion process):** `benign` vs `malignant`.
  Because malignant is always a lesion, "classifying normal vs malignant" is *subsumed by* lesion
  detection — you detect the lesion first, then split benign/malignant. Malignancy is therefore a
  **sub-task of the lesion hierarchy, never a separate pooled task.**

Default method for both levels = **general weight-transfer + full fine-tune (m1)**.

---

## 2. Per-dataset boundaries

### BCS-DBT (3D, former)
Four mutually-exclusive classes → mapped as:

| Original class | Maps to | Note |
|---|---|---|
| Normal | **normal** | no finding |
| Actionable | **DROPPED** | recalled, **not** biopsied → no confirmed/localized lesion; no ROI box exists. Permanently excluded. |
| Benign | **abnormal** | biopsied benign lesion (has ROI box) |
| Cancer | **abnormal** | malignant (has ROI box) |

- abnormal = `klass ∈ {benign, cancer}`; normal = `klass == 'normal'`.
- Rationale for dropping Actionable: it is neither a confirmed lesion nor a clean normal —
  it is an unresolved recall. Boxes in BCS-DBT exist **only** for benign+cancer, so
  Actionable has no lesion annotation to stand on.

### DBT-2026 (3D, new — Segmed cohort)
Label lives in the eCRF; every patient carries a Benign/Malignant `PathologyType`, so
`PathologyType` **cannot** be used directly (it would make everyone abnormal). The
**cohort Group** defines lesion presence:

| Group | Meaning | Maps to |
|---|---|---|
| A | biopsy-proven Malignant | **abnormal** |
| B | normal / benign screeners | **normal** (no-lesion) |
| C | Benign BI-RADS 0 callback | **abnormal** (benign lesion / finding) |
| D | biopsy-proven Benign | **abnormal** (benign lesion) |

- abnormal = `Group ∈ {A, C, D}`; normal = `Group == B`.
- Polygon annotations were considered but rejected as the label source: they don't cleanly
  separate classes (24 malignant cases lack polygons; 46 Group-B "normal" cases have them).
- One view-volume per `ROUTINE3D_VOL_{LCC,RCC,LMLO,RMLO}` series (single-frame
  V-Preview / Enhanced-Preview excluded). Patient-level label applied to all views
  (accepted v1 caveat: contralateral-breast label noise).

### 2D transfer-learning datasets — mapped to {normal, benign, malignant}
Each dataset is expressed in the three classes; the two task levels are then derived from it.
"has normal?" decides whether the set can do **Level-1 lesion detection**; "has both benign & malignant?"
decides **Level-2 malignancy**.

| Dataset | normal | benign | malignant | Level 1 (lesion)? | Level 2 (malignancy)? |
|---|---|---|---|---|---|
| **EMBED** | neg-assess & no ROI | ROI lesion (non-malig path) | ROI + malig path | ✅ default | ✅ (pathology) |
| **CDD-CESM** | Normal | Benign | Malignant | ✅ default | ✅ |
| **MIAS** | NORM | B (benign) | M (malignant) | ✅ default | ✅ |
| **CMMD** | — (none) | Benign | Malignant | ✗ (no normal) | ✅ only |
| **RSNA-BCD** | non-cancer | — (not labeled) | cancer | ✅ partial (normal vs malignant-lesion) | ✗ (no benign class) |
| INbreast | BI-RADS 1–2 | (BI-RADS 3 dropped) | BI-RADS 4–6 | ⚠ leakage → suppl. | ⚠ |
| BUSI | normal | benign | malignant | ✗ ultrasound (excluded) | ✗ |
| VinDr | BI-RADS 1–2 | — | BI-RADS 4–5 | ✗ no biopsy (dropped) | ✗ |

- **Default lesion-detection transfer set (Level 1):** EMBED, EMBED-natural, MIAS, CDD-CESM.
- **Malignancy division (Level 2):** CMMD, CDD-CESM, MIAS (EMBED where pathology available).
- RSNA contributes normal-vs-malignant only (no benign class) — a partial Level-1 with malignant-only positives.

---

## 3. Combined dataset (current) — normal : abnormal

| Split | BCS-DBT (actionable dropped) | DBT-2026 | Combined normal : abnormal |
|---|---|---|---|
| train | 18,432 (abn 200) | 1,558 (abn 1,302) | **18,488 : 1,502** |
| val | 1,003 (abn 75) | 383 (abn 305) | **1,006 : 380** |

- Training sampler: random **normal** undersampling to **normal:abnormal = 9:1**
  (13,518 : 1,502), epoch ≈ 15,020 volumes.
- Abnormal signal = **1,502** volumes (vs the former 76 malignant) — ~20× more positives;
  the primary reason for building the combined set.

---

## 4. Note (retired definition)

The earlier **malignant-only** positive (Cancer only; BCS-DBT 76 positives, normal:abnormal
≈ 99.4 : 0.6) is **retired** — it is superseded by the lesion-presence definition above and
should not be used for the combined pipeline.

---

## 5. Conventions

- **Ratio order:** always `normal : abnormal`.
- **Actionable:** permanently excluded from the combined pipeline.
- **Splits:** BCS-DBT uses its official splits; DBT-2026 uses deterministic `crc32(study_id)`
  (test 15% / val 15% / train 70%), independent of BCS-DBT.
- **Former files untouched:** all combined-pipeline code lives under `fine_tuning_combined/`;
  BCS-DBT lesion relabel is a subclass (`LesionBCSDBT`), not an edit to `fine_tuning/`.
