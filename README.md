# Diplomski rad — Analiza SMOTE algoritama

## Kontekst (za LLM agenta)

Ovo je diplomski rad na FERIT-u (Fakultet elektrotehnike, računarstva i informacijskih tehnologija Osijek). Tema: **analiza algoritama zasnovanih na SMOTE za rukovanje problemom neuravnoteženosti klasa**.

Rad se sastoji od:
- **LaTeX dokument** (`diplomski_rad.tex` + `poglavlja/`) — prva verzija s okvirnim planom sadržaja
- **Python implementacija** (`implementacija/`) — 11 SMOTE varijanti, evaluacijski okvir, web sučelje

## Struktura

```
.
├── diplomski_rad.tex          # Glavni LaTeX dokument
├── references.bib             # BibTeX literatura
├── poglavlja/                 # Poglavlja rada (.tex)
│   ├── 00-naslovnica.tex
│   ├── 00-sazetak-hr.tex
│   ├── 00-sazetak-en.tex
│   ├── 01-uvod.tex
│   ├── 02-neuravnotezenost-i-smote.tex
│   ├── 03-izvedenice-i-alternative.tex
│   ├── 04-programsko-rjesenje.tex
│   ├── 05-eksperimentalna-analiza.tex
│   ├── 06-web-sucelje.tex      # Integrirano u poglavlje 4
│   ├── 07-zakljucak.tex
│   └── 09-prilozi.tex
├── PLAN_IMPLEMENTACIJE.md     # Detaljan plan svih faza
│
└── implementacija/
    ├── config.py               # Centralna konfiguracija
    ├── requirements.txt        # Python ovisnosti
    ├── run_analysis.py         # Glavna skripta za 5.2 eksperimente
    ├── run_analysis_53.py      # Skripta za 5.3 eksperimente
    ├── compare_with_imblearn.py # Usporedba s referentnom bibliotekom
    │
    ├── smote_variants/         # 21 metoda resampliranja
    │   ├── base.py             # BaseSMOTE apstraktna klasa
    │   ├── smote.py            # SMOTE (Chawla 2002)
    │   ├── borderline.py       # Borderline-SMOTE BS1/BS2 (Han 2005)
    │   ├── adasyn.py           # ADASYN (He 2008)
    │   ├── safe_level.py       # Safe-Level-SMOTE (Bunkhumpornpat 2009)
    │   ├── kmeans_smote.py     # K-Means SMOTE (Douzas 2018)
    │   ├── svm_smote.py        # SVM-SMOTE (Nguyen 2011)
    │   ├── smote_enn.py        # SMOTE-ENN + SMOTE-Tomek (Batista 2004)
    │   ├── g_smote.py          # Geometric SMOTE (Douzas 2019)
    │   ├── random_smote.py     # Random SMOTE (Dong 2011)
    │   ├── polynom_fit.py      # Polynom-Fit SMOTE (Gazzah 2008)
    │   ├── baselines.py        # NoOversampling, RandomOver/Under-sampling
    │   ├── undersampling.py    # NearMiss-1/2/3, TomekLinks, ENN
    │   └── gan.py              # WGAN-GP generativno preuzorkovanje
    │
    ├── evaluation/             # Evaluacijski okvir
    │   ├── metrics.py          # 7 metrika (F1, G-Mean, AUC-ROC/PR, BA, MCC, F2)
    │   ├── cross_validator.py  # 5-fold stratificirani CV ×30 ponavljanja
    │   ├── experiment_runner.py # Orkestrator za 5.2 (grid svih kombinacija)
    │   └── experiment_runner_53.py # Orkestrator za 5.3 (fokusirani parovi)
    │
    ├── classifiers/defaults.py # 8 klasifikatora + _weighted varijante
    ├── data/generate_synthetic.py # Generator podataka + ugrađeni datasetovi
    │
    ├── analysis/               # Statistička analiza i vizualizacija
    │   ├── statistical.py      # Friedman, Nemenyi, Wilcoxon testovi
    │   └── vizualization.py    # CD dijagram, boxplot, heatmap
    │
    ├── web_app/                # FastAPI web sučelje za vizualizaciju
    │   ├── app.py              # Backend API + server
    │   └── index.html          # Frontend (dark tema, 3 moda: single/compare/all)
    │
    └── tests/                  # Testovi (90 passing + 5 WGAN skipped)
        ├── test_smote_variants.py
        ├── test_metrics.py
        └── test_pipeline.py
```

## Kako nastaviti rad

### Prvo pokretanje

```bash
pip install -r implementacija/requirements.txt
cd implementacija
python -m pytest tests/ -v        # Treba proći 51/51
python web_app/app.py              # Web sučelje na http://localhost:8501
```

### Web sučelje

Web app ima 3 moda (tabovi u sidebaru):
- **Pojedinačni** — jedan algoritam s opisom
- **Usporedba** — dva algoritma na identičnom skupu (zajednička PCA transformacija)
- **Svi (12)** — grid svih 12 varijanti na identičnom skupu

Podržani skupovi: slučajna klasifikacija, normalna/eksponencijalna/multimodalna distribucija, kružni podaci + 3 stvarna (breast cancer, wine, iris).

### Eksperimenti

```bash
python run_analysis.py SMOTE ADASYN    # Brzi test (2 varijante)
python run_analysis.py                  # Svi algoritmi, svi datasetovi
```

Rezultati se spremaju u `results/raw/` kao CSV.

### Validacija

- **51/51 testova prolazi** (`pytest tests/`)
- **Usporedba s imbalanced-learn**: 5/8 algoritama identičan broj sintetičkih uzoraka, 3/8 bliski
- Pipeline potvrđen end-to-end

## Što treba napraviti

### Prioritet: LaTeX rad
Sva poglavlja su trenutno u "prvoj verziji" — sadrže **planove/upute** što napisati (`"Opisat će se..."`, `"Navest će se..."`). Potrebno ih je pretvoriti u gotov tekst s:
- Stvarnim sadržajem umjesto uputa
- Ubačenim citatima (literatura je već pripremljena u `references.bib`)
- Slikama, tablicama, formulama

### Prioritet: Eksperimenti
1. Dodati **stvarne datasetove** u `data/real/` (trenutno su samo ugrađeni iz sklearn-a)
2. Pokrenuti `run_analysis.py` sa svim algoritmima, klasifikatorima i k-vrijednostima
3. Pokrenuti `run_analysis_53.py` za 5.3 usporedbu paradigmi
4. Generirati sve tablice i grafove za rad pomoću `analysis/` modula

### Gotovo (implementacija)
- ✅ 12 SMOTE varijanti (from-scratch)
- ✅ 5 undersampling metoda (NearMiss 1-3, TomekLinks, ENN)
- ✅ 3 baseline metode (NoOversampling, ROS, RUS)
- ✅ WGAN-GP generativno preuzorkovanje
- ✅ Classifier s class_weight='balanced' varijantama
- ✅ Dva eksperimentalna bloka (5.2 grid, 5.3 fokusirani)
- ✅ Web sučelje s 21 metodom

## Ključne napomene

- Svi SMOTE algoritmi su **from-scratch** implementacije prema originalnim radovima
- `imbalanced-learn` se koristi samo za validaciju i ugrađene datasetove, NE za SMOTE
- `sklearn.neighbors.NearestNeighbors` je jedina eksterna ovisnost za KNN pretragu
- Web sučelje koristi **FastAPI** (ne Streamlit) zbog kompatibilnosti s Python 3.14
- PCA/t-SNE u batch modu se računa **jednom** za sve algoritme — originalne točke su identične
- Broj poglavlja je smanjen sa 7 na 6 (web sučelje integrirano u poglavlje 4)
