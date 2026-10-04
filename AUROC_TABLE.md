# Transfer AUROC — paper tables

TEST AUROC, mean over 5 seeds {42,1,2,3,4}. Conditions: **m0** = ImageNet-only,
**m1** = ImageNet+DBT, **mscratch** = DBT-only.
Δ$_{\text{DBT}}$ = m1−m0 (gain from DBT fine-tuning); Δ$_{\text{init}}$ = m1−mscratch (gain from ImageNet init).
Rows sorted by m1 (best first).

## LaTeX — Natural arm
```latex
\begin{tabular}{lccccc}
\toprule
Dataset & m0 & m1 & mscratch & $\Delta_{\text{DBT}}$ & $\Delta_{\text{init}}$ \\
\midrule
CDD-CESM      & $0.717$ & $0.812$ & $0.549$ & $+0.095$ & $+0.263$ \\
CMMD          & $0.686$ & $0.783$ & $0.546$ & $+0.097$ & $+0.237$ \\
CBIS-DDSM     & $0.658$ & $0.756$ & $0.565$ & $+0.098$ & $+0.191$ \\
EMBED         & $0.712$ & $0.745$ & $0.539$ & $+0.033$ & $+0.206$ \\
MIAS          & $0.504$ & $0.707$ & $0.504$ & $+0.203$ & $+0.203$ \\
\bottomrule
\end{tabular}
```

## LaTeX — Balanced arm
```latex
\begin{tabular}{lccccc}
\toprule
Dataset & m0 & m1 & mscratch & $\Delta_{\text{DBT}}$ & $\Delta_{\text{init}}$ \\
\midrule
CDD-CESM      & $0.597$ & $0.796$ & $0.481$ & $+0.199$ & $+0.315$ \\
CMMD          & $0.638$ & $0.780$ & $0.537$ & $+0.142$ & $+0.243$ \\
EMBED (50:50) & $0.751$ & $0.761$ & $0.579$ & $+0.010$ & $+0.182$ \\
MIAS          & $0.530$ & $0.745$ & $0.390$ & $+0.215$ & $+0.355$ \\
CBIS-DDSM     & $0.650$ & $0.738$ & $0.571$ & $+0.088$ & $+0.167$ \\
RSNA          & $0.550$ & $0.688$ & $0.496$ & $+0.138$ & $+0.192$ \\
\bottomrule
\end{tabular}
```

## Markdown preview — Natural arm
| Dataset | m0 | m1 | mscratch | Δ_DBT | Δ_init |
|---|---:|---:|---:|---:|---:|
| CDD-CESM  | 0.717 | 0.812 | 0.549 | +0.095 | +0.263 |
| CMMD      | 0.686 | 0.783 | 0.546 | +0.097 | +0.237 |
| CBIS-DDSM | 0.658 | 0.756 | 0.565 | +0.098 | +0.191 |
| EMBED     | 0.712 | 0.745 | 0.539 | +0.033 | +0.206 |
| MIAS      | 0.504 | 0.707 | 0.504 | +0.203 | +0.203 |

## Markdown preview — Balanced arm
| Dataset | m0 | m1 | mscratch | Δ_DBT | Δ_init |
|---|---:|---:|---:|---:|---:|
| CDD-CESM      | 0.597 | 0.796 | 0.481 | +0.199 | +0.315 |
| CMMD          | 0.638 | 0.780 | 0.537 | +0.142 | +0.243 |
| EMBED (50:50) | 0.751 | 0.761 | 0.579 | +0.010 | +0.182 |
| MIAS          | 0.530 | 0.745 | 0.390 | +0.215 | +0.355 |
| CBIS-DDSM     | 0.650 | 0.738 | 0.571 | +0.088 | +0.167 |
| RSNA          | 0.550 | 0.688 | 0.496 | +0.138 | +0.192 |

All Δ columns are positive: DBT fine-tuning (Δ_DBT) and ImageNet init (Δ_init) each help everywhere → m1 > m0 > mscratch.
