# Plan implementacije — Analiza SMOTE algoritama za neuravnotežene klase

---

## 1. Struktura projekta

```
projekt/
├── run_analysis.py              # Glavna skripta za pokretanje eksperimenata
├── run_validation.py            # Skripta za pokretanje testova validacije
├── requirements.txt             # Popis Python ovisnosti
├── config.py                    # Centralna konfiguracija (parametri, putanje)
│
├── smote_variants/              # Paket s implementacijama SMOTE algoritama
│   ├── __init__.py              # Izvoz svih varijanti
│   ├── base.py                  # BaseSMOTE — apstraktna bazna klasa
│   ├── smote.py                 # SMOTE (Chawla 2002)
│   ├── borderline.py            # Borderline-SMOTE (BS1 i BS2, Han 2005)
│   ├── adasyn.py                # ADASYN (He 2008)
│   ├── safe_level.py            # Safe-Level-SMOTE (Bunkhumpornpat 2009)
│   ├── kmeans_smote.py          # K-Means SMOTE (Douzas 2018)
│   ├── svm_smote.py             # SVM-SMOTE (Nguyen 2011)
│   ├── smote_enn.py             # SMOTE-ENN (Batista 2004)
│   ├── smote_tomek.py           # SMOTE-Tomek (Batista 2004)
│   ├── g_smote.py               # Geometric SMOTE (Douzas 2019)
│   ├── random_smote.py          # Random SMOTE (Dong 2011)
│   └── polynom_fit.py           # Polynom-Fit SMOTE (Gazzah 2008)
│
├── evaluation/                  # Evaluacijski okvir
│   ├── __init__.py
│   ├── cross_validator.py       # Stratificirana 5-fold CV + ponavljanja
│   ├── metrics.py               # Sve metrike: F1, G-Mean, AUC-ROC/PR itd.
│   └── experiment_runner.py     # Orkestriranje: SMOTE → CV → klasifikator → metrike
│
├── classifiers/                 # Konfiguracija klasifikatora
│   ├── __init__.py
│   └── defaults.py              # Svih 8 klasifikatora s default parametrima
│
├── data/                        # Podaci
│   ├── real/                    # Stvarni skupovi (CSV)
│   ├── synthetic/               # Sintetički skupovi (CSV)
│   └── generate_synthetic.py    # Generator sintetičkih skupova
│
├── analysis/                    # Statistička analiza i vizualizacija
│   ├── __init__.py
│   ├── statistical.py           # Friedman, Nemenyi, Wilcoxon testovi
│   └── vizualization.py         # CD dijagrami, box-plot, heatmap
│
├── web_app/                     # Streamlit web sučelje
│   ├── app.py                   # Glavna Streamlit aplikacija
│   ├── pages/                   # Podstranice (ako ih bude)
│   └── utils.py                 # Pomoćne funkcije za web (PCA, t-SNE)
│
├── tests/                       # Jedinični i integracijski testovi
│   ├── test_smote_variants.py   # Testovi za svaki SMOTE algoritam
│   ├── test_metrics.py          # Testovi za evaluacijske metrike
│   └── test_pipeline.py         # End-to-end test jedne kombinacije
│
├── results/                     # Izlazni rezultati (kreira se pri pokretanju)
│   ├── raw/                     # CSV datoteke sa sirovim rezultatima
│   ├── tables/                  # Generirane tablice (LaTeX-ready)
│   └── figures/                 # Generirani grafovi (PDF/PNG)
│
├── diplomski_rad.tex            # Glavni LaTeX dokument
├── references.bib               # BibTeX literatura
└── poglavlja/                   # Poglavlja rada (.tex)
```

---

## 2. Detaljan opis modula

### 2.1. `config.py`

Centralna konfiguracija cijelog projekta — SVI moduli čitaju odavde, nigdje hardkodirano.

```python
SMOTE_K_VALUES = [3, 5, 7, 10]
CV_FOLDS = 5
CV_REPEATS = 30
RANDOM_SEEDS = range(42, 42 + CV_REPEATS)
METRICS = ["f1", "g_mean", "auc_roc", "auc_pr", "balanced_accuracy", "mcc", "f2"]
CLASSIFIERS = ["dt", "rf", "xgboost", "lr", "svm", "knn", "gnb", "mlp"]
DATASET_DIRS = {"real": "data/real", "synthetic": "data/synthetic"}
RESULTS_DIR = "results"
```

---

### 2.2. `smote_variants/` — SRCE PROJEKTA

#### `base.py` — BaseSMOTE

```python
from abc import ABC, abstractmethod

class BaseSMOTE(ABC):
    def __init__(self, k=5, sampling_strategy="auto", random_state=None):
        self.k = k
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    @abstractmethod
    def fit_resample(self, X, y):
        """Vraća (X_resampled, y_resampled)."""
        pass

    def _validate_input(self, X, y):
        """Provjera: 2D numpy, binarne klase, barem 2 manjinska primjera."""
        pass
```

**Validacija:** Svaka izvedenica prolazi (1) test dimenzija izlaza, (2) test tipova, (3) test da se manjinska klasa povećala za traženi broj.

---

#### `smote.py` — SMOTE (Chawla 2002)

- Za svaki $x_i$ u manjinskoj klasi: $k$-NN (Euklidska udaljenost) unutar manjinske klase.
- Slučajni susjed $x_{nn}$, $\lambda \sim U(0,1)$, $x_{new} = x_i + \lambda \cdot (x_{nn} - x_i)$.
- Ponavlja se do balansiranja.
- Parametri: `k`, `sampling_strategy`.
- Oslanja se na `sklearn.neighbors.NearestNeighbors`.

**Validacija:**
1. Jedinični test na 2D toy skupu (npr. `make_classification` s IR=0.1): provjera da je broj manjinskih = broj većinskih nakon resampliranja.
2. Test da sintetičke točke leže na segmentima između originalnih manjinskih primjera.
3. Test s `k=1` — sintetičke točke moraju biti identične originalnima (jer je $\lambda$ jedini susjed).

---

#### `borderline.py` — Borderline-SMOTE (Han 2005)

- Za svaki manjinski primjer: broj većinskih među $m$ susjeda → Safe / Danger / Noise.
- BS1: preuzorkuje Danger, koristi samo manjinske susjede.
- BS2: preuzorkuje Danger, dopušta većinske susjede uz $\lambda \in (0, 0.5)$.
- Parametri: `k`, `m`, `kind` (BS1/BS2).

**Validacija:**
1. Jedinični test da se Safe primjeri nikad ne preuzorkuju.
2. Test da BS2 generira bliže manjinskoj strani ($\lambda < 0.5$).

---

#### `adasyn.py` — ADASYN (He 2008)

- $\Gamma_i = \Delta_i / k$, gdje je $\Delta_i$ broj većinskih susjeda.
- Normalizacija $\hat{\Gamma}_i$, ukupno $G$ novih primjera raspodijeljeno proporcionalno.
- Parametri: `k`, `sampling_strategy`.

**Validacija:**
1. Test da područja s više većinskih susjeda dobiju više sintetičkih primjera.

---

#### `safe_level.py` — Safe-Level-SMOTE (Bunkhumpornpat 2009)

- Sigurnosna razina $sl_p$ = broj manjinskih susjeda među $k$.
- Strategija ovisno o ($sl_p$, $sl_n$).
- Parametri: `k`.

**Validacija:**
1. Test da se nikad ne generiraju točke kad je $sl_p = sl_n = 0$.

---

#### `kmeans_smote.py` — K-Means SMOTE (Douzas 2018)

- K-Means na manjinskoj klasi → identifikacija rijetkih klastera.
- Težina $\propto 1/|klaster|$ → više primjera u rijetkim klasterima.
- SMOTE unutar svakog klastera.
- Parametri: `k`, `n_clusters`.

**Validacija:**
1. Test da rijetki klasteri dobiju proporcionalno više primjera.

---

#### `svm_smote.py` — SVM-SMOTE (Nguyen 2011)

- Treniranje SVM-a na originalnim podacima.
- Ekstrakcija potpornih vektora manjinske klase.
- SMOTE samo na njima.
- Parametri: `k`, `svm_params`.

**Validacija:**
1. Test da se preuzorkuju samo potporni vektori.

---

#### `smote_enn.py` i `smote_tomek.py` (Batista 2004)

- SMOTE → ENN čišćenje (ukloni ako većina $k$-NN krivo klasificira).
- SMOTE → Tomek Links čišćenje.
- Parametri: `k` (za SMOTE), `k_enn`.

**Validacija:**
1. Test da je konačan broj primjera manji ili jednak nego nakon samog SMOTE-a.

---

#### Geometrijska proširenja

| Datoteka | Algoritam | Ključna razlika |
|----------|-----------|-----------------|
| `g_smote.py` | G-SMOTE (Douzas 2019) | Generira unutar sektora, ne samo na liniji. Parametri: `k`, `alpha`, `truncation_factor` |
| `random_smote.py` | Random SMOTE (Dong 2011) | Slučajni smjer i udaljenost. Parametar: `k` |
| `polynom_fit.py` | Polynom-Fit (Gazzah 2008) | Polinomna interpolacija kroz više susjeda. Parametar: `k` |

**Validacija (za sva tri):**
1. Test da generirane točke NISU sve na linijskim segmentima (za razliku od osnovnog SMOTE).
2. Test dimenzija i tipova.

---

### 2.3. `evaluation/` — EVALUACIJSKI OKVIR

#### `cross_validator.py`

```python
def stratified_repeated_cv(X, y, n_splits=5, n_repeats=30, seeds=None):
    """Generator koji vraća (X_train, X_test, y_train, y_test) za svaku iteraciju."""
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(X, y):
            yield X[train_idx], X[test_idx], y[train_idx], y[test_idx]
```

**Validacija:** Test da je omjer klasa očuvan u svakom foldu (unutar tolerancije ±5%).

---

#### `metrics.py`

Sve metrike implementirane pozivom `sklearn.metrics`:
- `f1_score(y_true, y_pred)`
- `balanced_accuracy_score(y_true, y_pred)`
- `roc_auc_score(y_true, y_score)`
- `average_precision_score(y_true, y_score)` (AUC-PR)
- `matthews_corrcoef(y_true, y_pred)` (MCC)
- `fbeta_score(y_true, y_pred, beta=2)` (F2)
- G-Mean = $\sqrt{\text{sensitivity} \cdot \text{specificity}}$ (ručno)

**Validacija:**
1. Test na poznatom ulazu: savršen klasifikator → sve metrike = 1.0.
2. Test na nasumičnom klasifikatoru (50/50) → metrike blizu 0.5 / 0.0 (ovisno o metrici).
3. Ekstremni disbalans test: 99 većinskih, 1 manjinski → accuracy = 0.99, ali F1 ≈ 0. Potvrda da accuracy nije pouzdan.

---

#### `experiment_runner.py`

Glavni orkestrator — za svaku kombinaciju (dataset, SMOTE, k, classifier) poziva:

```python
for dataset_name, (X, y) in datasets:
    for smote_name, smote_cls in SMOTE_VARIANTS.items():
        for k in SMOTE_K_VALUES:
            smote = smote_cls(k=k)
            for classifier_name, clf in classifiers.items():
                results = run_cv(X, y, smote, clf, cv_generator)
                save_results(results, dataset_name, smote_name, k, classifier_name)
```

Bazne metode (bez preuzorkovanja, random oversampling, random undersampling) također se evaluiraju u istom okviru.

**Validacija (end-to-end):**
1. Pokreni na 1 toy skupu sa SVM klasifikatorom → potvrdi da se rezultati zapisuju u CSV.
2. Potvrdi da su sve metrike u intervalu [0, 1].
3. Potvrdi da random oversampling daje iste ili bolje rezultate od baselinea (bez preuzorkovanja).

---

### 2.4. `classifiers/defaults.py`

Svih 8 klasifikatora s fiksnim parametrima:

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

CLASSIFIERS = {
    "dt": DecisionTreeClassifier(random_state=42),
    "rf": RandomForestClassifier(n_estimators=100, random_state=42),
    "xgboost": XGBClassifier(scale_pos_weight="auto", random_state=42, use_label_encoder=False, eval_metric="logloss"),
    "lr": LogisticRegression(max_iter=1000, random_state=42),
    "svm": SVC(kernel="rbf", probability=True, random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "gnb": GaussianNB(),
    "mlp": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
}
```

**Validacija:** Svaki klasifikator se može instancirati i trenirati na malom toy skupu bez greške.

---

### 2.5. `data/generate_synthetic.py`

Generator 3 vrste sintetičkih skupova:

1. **`make_classification`** (sklearn) — kontrolirani IR, dimenzionalnost, šum.
2. **Distribucijski** — normalna, eksponencijalna, multimodalna raspodjela po klasama.
3. **Geometrijski** — klase s definiranim preklapanjem/manifold strukturom.

Parametri za svaki skup: IR (1.5–50), $m$ (2–100), šum (0–30%), $n$ (200–5000).

**Validacija:**
1. Generirani skup ima točan IR.
2. Sve značajke su realni brojevi.
3. Točno dvije klase.

---

### 2.6. `analysis/statistical.py`

```python
from scipy.stats import friedmanchisquare, wilcoxon
from scikit_posthocs import posthoc_nemenyi_friedman

def friedman_test(rankings):
    """Vraća statistiku i p-vrijednost."""

def nemenyi_posthoc(rankings):
    """Vraća matricu p-vrijednosti i CD (critical difference)."""

def wilcoxon_vs_baseline(results, baseline="SMOTE"):
    """Uspoređuje svaku izvedenicu s baseline SMOTE-om."""
```

**Validacija:**
1. Test na ručno konstruiranim rankingsima s poznatim ishodom.
2. Friedman na identičnim rankingsima → p > 0.05 (nema razlike).
3. Friedman na drastično različitim rankingsima → p < 0.05.

---

### 2.7. `analysis/vizualization.py`

Funkcije za generiranje grafova (spremaju se u `results/figures/`):

```python
def plot_cd_diagram(rankings, metric_name):
    """Kritični dijagram (matplotlib + Orange CD diagram)."""

def plot_boxplot(results_df, metric_name):
    """Box-plot svih algoritama za jednu metriku."""

def plot_heatmap(results_df, metric_name):
    """Heatmap: algoritmi × skupovi podataka."""

def plot_synthetic_scatter(X_orig, y_orig, X_synth, y_synth):
    """Scatter originalnih i sintetičkih primjera (2D PCA)."""
```

---

### 2.8. `web_app/` — STREAMLIT SUČELJE

#### `app.py`

Layout:
1. Sidebar: upload CSV ili odabir ugrađenog skupa.
2. Sidebar: odabir SMOTE varijante + parametri ($k$, postotak).
3. Main: Plotly scatter (PCA/t-SNE redukcija, 2D/3D).
4. Main: granica odluke (SVM ili RF) prije i poslije SMOTE-a.
5. Main: tablica metrika za odabrane varijante.

#### `utils.py`

```python
def reduce_dimensions(X, method="pca", n_components=2):
    """PCA ili t-SNE redukcija."""

def compute_decision_boundary(clf, X, y, grid_resolution=200):
    """Računa grid za prikaz granice odluke u 2D."""

def get_metrics_table(smote_results):
    """Iz rezultata računa tablicu metrika za prikaz."""
```

---

### 2.9. `tests/`

#### `test_smote_variants.py`

Za svaki od 11 algoritama:
- `test_output_shape`: provjera dimenzija izlaza.
- `test_minority_increased`: manjinska klasa se povećala.
- `test_reproducibility`: isti seed → isti izlaz.
- `test_no_nan_inf`: nema NaN/Inf vrijednosti u izlazu.

#### `test_metrics.py`

- `test_perfect_classifier`: sve metrike = 1.0 (ili blizu).
- `test_worst_classifier`: sve metrike = 0.0.
- `test_imbalanced_accuracy_trap`: accuracy visok, F1 nizak.

#### `test_pipeline.py`

- `test_full_pipeline_one_combination`: SMOTE → CV → 1 klasifikator → CSV output.
- `test_all_smote_instantiate`: svih 11 algoritama se može instancirati bez greške.

---

### 2.10. `requirements.txt`

```
numpy>=1.21
scipy>=1.7
scikit-learn>=1.0
imbalanced-learn>=0.9
xgboost>=1.5
pandas>=1.3
matplotlib>=3.4
seaborn>=0.11
scikit-posthocs>=0.7
streamlit>=1.10
plotly>=5.0
pytest>=7.0
```

---

## 3. Redoslijed implementacije

### Faza 1 — Jezgra (1.–2. tjedan)
1. `config.py` + `requirements.txt`
2. `smote_variants/base.py` — apstraktna klasa + validacija ulaza
3. `smote_variants/smote.py` — osnovni SMOTE
4. `tests/test_smote_variants.py` — prvi testovi za SMOTE
5. **Validacijska točka:** `python -m pytest tests/test_smote_variants.py -v` — svi testovi prolaze za SMOTE.

### Faza 2 — Sve SMOTE varijante (2.–3. tjedan)
6. Redom: `borderline.py`, `adasyn.py`, `safe_level.py`, `kmeans_smote.py`, `svm_smote.py`
7. `smote_enn.py`, `smote_tomek.py`
8. `g_smote.py`, `random_smote.py`, `polynom_fit.py`
9. Proširiti `tests/test_smote_variants.py` za svaku novu varijantu.
10. **Validacijska točka:** `pytest` — svih 11 algoritama prolazi jedinične testove.

### Faza 3 — Evaluacijski okvir (3.–4. tjedan)
11. `evaluation/metrics.py` + `test_metrics.py`
12. `evaluation/cross_validator.py`
13. `classifiers/defaults.py`
14. `evaluation/experiment_runner.py`
15. `tests/test_pipeline.py`
16. **Validacijska točka:** Pokrenuti `test_pipeline.py` — potvrda da jedna kombinacija (SMOTE + RF + toy skup) daje validne CSV rezultate.

### Faza 4 — Podaci i eksperimenti (4.–5. tjedan)
17. `data/generate_synthetic.py`
18. Skupiti stvarne skupove u `data/real/`
19. `run_analysis.py` — glavna skripta
20. **Validacijska točka:** Pokrenuti `run_analysis.py` na 1 stvarnom + 1 sintetičkom skupu, potvrditi CSV output u `results/raw/`.

### Faza 5 — Analiza i vizualizacija (5.–6. tjedan)
21. `analysis/statistical.py`
22. `analysis/vizualization.py`
23. **Validacijska točka:** Pokrenuti statističke testove na stvarnim rezultatima — potvrditi da se generiraju CD dijagrami.

### Faza 6 — Streamlit sučelje (6.–7. tjedan)
24. `web_app/utils.py`
25. `web_app/app.py`
26. **Validacijska točka:** `streamlit run web_app/app.py` — sučelje radi, učitava skup, prikazuje PCA i SMOTE rezultate.

### Faza 7 — Potpuni eksperimenti (7. tjedan)
27. Pokretanje `run_analysis.py` za SVE kombinacije.
28. Generiranje SVIH tablica i grafova za rad.
29. **Validacijska točka:** Provjera da su svi očekivani outputi prisutni u `results/`.

---

## 4. Ključne validacijske točke — sažeti popis

| # | Što se provjerava | Kako |
|---|---|---|
| V1 | Osnovni SMOTE daje ispravan broj primjera | Jedinični test na toy skupu |
| V2 | Svih 11 varijanti se instancira bez greške | `pytest test_smote_variants.py` |
| V3 | Metrike daju ispravne vrijednosti za poznate slučajeve | `pytest test_metrics.py` |
| V4 | Pipeline od podataka do CSV-a radi | `pytest test_pipeline.py` |
| V5 | Sintetički skupovi imaju točan IR i dimenzije | `generate_synthetic.py` + ručna provjera |
| V6 | Rezultati su deterministički (isti seed = isti output) | Jedinični test s fiksnim seedom |
| V7 | Nema NaN/Inf u izlazima niti jednog algoritma | Test za svaki SMOTE |
| V8 | Stratifikacija čuva omjer klasa u CV foldu | Test cross_validatora |
| V9 | Friedmanov test detektira razlike tamo gdje postoje | Test na ručnim rankingsima |
| V10 | Streamlit se pokreće i prikazuje osnovne funkcionalnosti | Ručno testiranje |

---

## 5. Očekivani output nakon potpunog pokretanja

```
results/
├── raw/
│   ├── results_f1.csv            # F1 za sve kombinacije
│   ├── results_g_mean.csv        # G-Mean za sve kombinacije
│   ├── results_auc_roc.csv
│   ├── results_auc_pr.csv
│   ├── results_balanced_accuracy.csv
│   ├── results_mcc.csv
│   └── results_f2.csv
├── tables/
│   ├── table_avg_rankings.tex    # Tablica prosječnih rangova
│   └── table_p_values.tex        # Tablica p-vrijednosti stat. testova
└── figures/
    ├── cd_diagram_f1.pdf         # CD dijagram za F1
    ├── cd_diagram_auc_roc.pdf    # CD dijagram za AUC-ROC
    ├── cd_diagram_mcc.pdf        # CD dijagram za MCC
    ├── boxplot_f1.pdf
    ├── boxplot_auc_roc.pdf
    ├── heatmap_alg_dataset.pdf
    └── scatter_synthetic_*.pdf   # Scatter prikazi za web sučelje
```
