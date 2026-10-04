# Transfer-Learning Data Splits

Per-split composition. Ratio = neg : pos. All splits patient-grouped (leakage-safe), deterministic (crc32 / greedy).
Positive: lesion task = abnormal (benign+malignant); malignancy task = malignant (RSNA = cancer).

| dataset | task | split unit | split | n | neg : pos | pos % |
|---|---|---|---|---|---|---|
| **EMBED-natural** | lesion | patient (empi_anon) | train | 7,717 | 4,900 : 2,817 | 36.5 |
| | | | val | 1,654 | 1,050 : 604 | 36.5 |
| | | | test | 1,654 | 1,050 : 604 | 36.5 |
| **EMBED (50:50)** | lesion | patient (empi_anon) | train | 9,800 | 4,900 : 4,900 | 50.0 |
| | | | val | 2,100 | 1,050 : 1,050 | 50.0 |
| | | | test | 2,100 | 1,050 : 1,050 | 50.0 |
| **MIAS** | lesion | patient (L/R pair) | train | 226 | 145 : 81 | 35.8 |
| | | | val | 48 | 31 : 17 | 35.4 |
| | | | test | 48 | 31 : 17 | 35.4 |
| **CDD-CESM** | lesion | patient (Patient_ID) | train | 665 | 218 : 447 | 67.2 |
| | | | val | 91 | 30 : 61 | 67.0 |
| | | | test | 247 | 93 : 154 | 62.3 |
| **CMMD** | malignancy | patient (PatientID) | train | 2,636 | 806 : 1,830 | 69.4 |
| | | | val | 358 | 106 : 252 | 70.4 |
| | | | test | 746 | 198 : 548 | 73.5 |
| **RSNA** | malignancy | patient/site | train | 38,322 | 37,479 : 843 | 2.2 |
| | | | val | 5,479 | 5,353 : 126 | 2.3 |
| | | | test | 10,867 | 10,680 : 187 | 1.7 |
| **CBIS-DDSM** | malignancy | patient (patient_id) | train | 1,943 | 1,067 : 876 | 45.1 |
| | | | val | 309 | 183 : 126 | 40.8 |
| | | | test | 605 | 330 : 275 | 45.5 |

Split fractions: 70 / 15 / 15 (patient-grouped). EMBED uses a greedy image-balanced patient split; others crc32(patient) deterministic.
Notes: EMBED-natural = fixed-seed random 36.5% subsample of the 50:50 subset (test kept natural). CMMD/CDD-CESM natural ratios are pos-majority. RSNA ~2% (screening prevalence). The neg:pos above is the *natural* composition; the balanced arm undersamples the training majority to 1:1 (val/test unchanged).
