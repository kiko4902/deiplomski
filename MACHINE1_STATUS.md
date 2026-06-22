# Machine 1 (PC) — Status & Plan

## Što je gotovo

| Što | Detalji |
|-----|---------|
| **Glavni run** | 20 metoda × 6 klasifikatora (rf,lr,dt,knn,gnb,xgboost) × 18 datasetova × k=[3,5] × 10 CV repeats |
| **Izlaz** | 18 CSV-ova u F:\results\raw\ (svaki 70-125 KB) |
| **Statistička analiza** | Friedman + Nemenyi + Wilcoxon — gotovo |
| **Vizualizacija** | 15 PDF grafova (boxplot, violin, CD, heatmap, bars × 3 metrike) |
| **LaTeX tablice** | avg_rankings, nemenyi_pvalues, wilcoxon_vs_baseline, wilcoxon_vs_nooversampling |
| **k-test** | Potvrđeno: k=7,10 ne donosi značajnu razliku — 5 datasetova testirano |

## Što pokrenuti u preostalih ~36 sati

### Prioritet 1: Full run (preko noći, ~8-12h)

```powershell
cd "C:\Users\Kristian\Desktop\New folder\implementacija"
python run_full.py 2>&1 | Tee-Object -FilePath "run_full_log.txt"
```

**Što radi:** SVIH 8 klasifikatora (uklj. SVM, MLP), k=[3,5,7], CV_REPEATS=30, SVHI datasetovi.
Ovo daje najrigoroznije rezultate — idealno za pokazati mentoru.

### Prioritet 2: Analiza full runa (nakon što završi, ~15 min)

```powershell
python -c "import sys; sys.stdout.reconfigure(encoding='utf-8'); from analysis.statistical import run_all_statistical_tests; from analysis.vizualization import generate_all_plots; [run_all_statistical_tests(m) or generate_all_plots(m) for m in ['f1','g_mean','auc_roc']]"
```

### Prioritet 3: Ako ostane vremena — analyze_results.py

```powershell
python analyze_results.py
```

Daje brzi pregled: baseline vs SMOTE po datasetu, ranking, korelacija metrika, grupe metoda.

## Trenutni ključni nalazi

- **Friedman p < 0.001** za F1, G-Mean, AUC-ROC — metode se značajno razlikuju
- **G-SMOTE** i **SMOTE-Tomek** jedine su konzistentno ravne osnovnom SMOTE-u
- **SMOTE poboljšava G-Mean univerzalno** (p<0.001 vs NoOversampling)
- **SMOTE ne poboljšava F1 univerzalno** (p=0.77 vs NoOversampling) — pomaže samo na nekim datasetovima
- **NearMiss poduzorkovanje je konzistentno lošije** od SMOTE varijanti
- **k=7,10 ne donosi razliku** vs k=3,5

## Putanje

- Rezultati: F:\results\
- Kod: C:\Users\Kristian\Desktop\New folder\implementacija\
- Git: https://github.com/kiko4902/deiplomski
