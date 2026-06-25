# Diplomski rad — Status Report za mentora

## 1. Što je napravljeno

### Implementacija

20 metoda resamplinga implementirano from-scratch u Pythonu:

| Kategorija | Algoritmi | Broj |
|-----------|-----------|------|
| SMOTE varijante | SMOTE, Borderline-SMOTE (BS1, BS2), ADASYN, SafeLevel-SMOTE, KMeans-SMOTE, SVM-SMOTE, SMOTE-ENN, SMOTE-Tomek, G-SMOTE, Random-SMOTE, PolynomFit-SMOTE | 12 |
| Undersampling | NearMiss-1, NearMiss-2, NearMiss-3, Tomek Links, Edited Nearest Neighbors (ENN) | 5 |
| Baseline | NoOversampling, RandomOversampling, RandomUndersampling | 3 |

Sve SMOTE implementacije su from-scratch, pisane prema originalnim radovima. Biblioteka `imbalanced-learn` koristi se isključivo za validaciju (usporedba rezultata) i učitavanje datasetova.

**Validacija implementacije:**
- 90/95 testova prolazi (5 WGAN testova preskočeno — nema PyTorch)
- Usporedba s imbalanced-learn: 5/8 algoritama daje identičan broj sintetičkih uzoraka
- Reproducibilnost: isti seed → identični rezultati
- Web sučelje (FastAPI + Plotly) za interaktivnu vizualizaciju svih varijanti

### Eksperimentalna postavka

| Parametar | Vrijednost | Obrazloženje |
|-----------|-----------|--------------|
| k (broj susjeda) | 3, 5, 7 | Chawla 2002 preporučuje k=5; testirano i 3 i 7 |
| CV | 5-fold stratificirani × 10 (×30 za full run) | Standard u literaturi (Demšar 2006) |
| Klasifikatori | DT, RF, XGBoost, LR, SVM, kNN, GNB, MLP | 8 klasifikatora različitih familija |
| Metrike | F1, G-Mean, AUC-ROC, AUC-PR, Balanced Accuracy, MCC, F2 | 7 metrika |
| Skupovi | 11 stvarnih + 7 sintetičkih | IR 1.7–28.1, d 3–100 |
| Statistika | Friedman + Nemenyi + Wilcoxon | Demšar 2006 metodologija |
| Ukupno | 20 metode × 8 klas. × 18 dataseta × 7 metrika = 54,432 reda | Full run gotov |

---

## 2. Opis svakog algoritma

### 2.1. SMOTE — Synthetic Minority Over-sampling Technique (Chawla 2002)

**Osnovna ideja:** Umjesto dupliciranja postojećih manjinskih primjera (što dovodi do overfittinga), SMOTE generira nove, sintetičke primjere linearnom interpolacijom između postojećih manjinskih uzoraka.

**Kako funkcionira:** Za svaki primjer manjinske klase pronalazi se k najbližih susjeda (Euklidska udaljenost). Slučajno se bira jedan susjed i generira novi primjer na dužini koja ih spaja: x_new = x_i + λ(x_nn − x_i), gdje je λ ~ U(0,1). Postupak se ponavlja dok se klase ne izbalansiraju.

**Zaključak iz eksperimenta:** SMOTE je robustan i konzistentan — rank #4 globalno za F1 (0.7493). Niti jedna varijanta ga ne nadmašuje univerzalno, iako specifične varijante pobjeđuju u specifičnim scenarijima.

---

### 2.2. Borderline-SMOTE (Han 2005)

**Osnovna ideja:** Nisu svi manjinski primjeri jednako važni — oni na granici odluke su ključni. Algoritam se fokusira samo na "opasne" (danger) primjere.

**Kako funkcionira:** Svaki manjinski primjer kategorizira se prema broju većinskih susjeda: Safe, Danger ili Noise. BS1 preuzorkuje samo Danger primjere koristeći manjinske susjede. BS2 dopušta i većinske susjede ali s λ ∈ (0, 0.5) — generira bliže manjinskoj strani.

**Zaključak:** Rank #12-13 globalno. Lošiji od SMOTE-a na većini datasetova (p<0.001), ali može pomoći na specifičnim skupovima s jasnom granicom (wine_quality: +0.006 vs SMOTE).

---

### 2.3. ADASYN — Adaptive Synthetic Sampling (He 2008)

**Osnovna ideja:** Generirati više sintetičkih primjera u područjima gdje je većinska klasa gušća — tamo gdje je klasifikacija najteža.

**Kako funkcionira:** Računa se omjer Γ_i = broj većinskih susjeda / k za svaki manjinski primjer. Normalizira se u distribuciju i ukupni broj sintetičkih primjera se raspoređuje proporcionalno — više uzoraka u "teškim" područjima.

**Zaključak:** Rank #11 globalno. Konzistentno lošiji od SMOTE-a (p<0.001). Problem: outlieri dobivaju najviše sintetičkih primjera, pojačavajući šum.

---

### 2.4. SafeLevel-SMOTE (Bunkhumpornpat 2009)

**Osnovna ideja:** Generirati sintetičke primjere samo u "sigurnim" područjima gdje ima dovoljno manjinskih susjeda — izbjeći generiranje u šumu.

**Kako funkcionira:** Sigurnosna razina (safe-level) = broj manjinskih susjeda među k najbližih. Novi primjer pozicionira se bliže onom s višom sigurnosnom razinom. Ako su oba u nesigurnom području, generiranje se preskače.

**Zaključak:** Rank #3 globalno — iznenađujuće dobar! Jedna od rijetkih varijanti koje su konzistentno u vrhu. Konzervativan pristup se isplati — izbjegavanje šuma daje stabilne rezultate.

---

### 2.5. KMeans-SMOTE (Douzas 2018)

**Osnovna ideja:** Riješiti problem unutar-klasnog disbalansa — manjinska klasa često nije homogena, nego se sastoji od manjih podgrupa (klastera).

**Kako funkcionira:** K-Means grupira manjinske primjere. Rijetki klasteri dobivaju proporcionalno više sintetičkih primjera (težinski ~ 1/veličina_klastera). SMOTE se primjenjuje unutar svakog klastera zasebno.

**Zaključak:** Rank #8 globalno. Bolji od SMOTE-a na high-IR datasetovima (yeast_me2: +0.011, us_crime: +0.017). Posebno koristan kad manjinska klasa ima više podgrupa.

---

### 2.6. SVM-SMOTE (Nguyen 2011)

**Osnovna ideja:** Generirati sintetičke primjere samo oko potpornih vektora — točaka koje definiraju granicu odluke.

**Kako funkcionira:** Trenira se SVM na originalnim podacima. Izdvajaju se potporni vektori manjinske klase. Novi primjeri generiraju se duž linija koje spajaju potporne vektore s njihovim susjedima.

**Zaključak:** Rank #7 globalno. Sličan SMOTE-u ukupno, ali zahtijeva treniranje SVM-a (dodatni trošak). Koristan kad je granica odluke jasna.

---

### 2.7. SMOTE-ENN (Batista 2004)

**Osnovna ideja:** Kombinirati preuzorkovanje s naknadnim čišćenjem — prvo generirati, pa ukloniti šumne primjere.

**Kako funkcionira:** SMOTE → ENN (Edited Nearest Neighbors): uklanja svaki primjer kojeg većina njegovih k-NN susjeda pogrešno klasificira. Agresivno čišćenje.

**Zaključak:** Rank #10 globalno. Bolji od SMOTE-a na abalone (+0.027), ali generalno agresivno brisanje smanjuje performanse.

---

### 2.8. SMOTE-Tomek (Batista 2004)

**Osnovna ideja:** SMOTE + uklanjanje Tomek Link parova — blago čišćenje granice odluke.

**Kako funkcionira:** SMOTE → uklanjanje Tomek Linkova (parova različitih klasa koji su međusobno najbliži susjedi). Manje agresivno od ENN-a.

**Zaključak:** Rank #2 globalno! Jedna od najboljih varijanti. Konzistentno u top 3 kroz sve eksperimente. Blago čišćenje + SMOTE = pouzdana kombinacija.

---

### 2.9. G-SMOTE — Geometric SMOTE (Douzas 2019)

**Osnovna ideja:** Osnovni SMOTE generira samo na liniji između dvije točke — G-SMOTE proširuje generiranje na cijeli geometrijski sektor.

**Kako funkcionira:** Generira unutar višedimenzionalnog sektora definiranog manjinskim primjerom i smjerom prema susjedu. Kontrolira se faktorom deformacije α (udaljenost od linije) i faktorom skraćivanja. Veća raznolikost sintetičkih uzoraka.

**Zaključak:** Rank #9 globalno. Teorijski najnaprednija varijanta, ali u praksi ne nadmašuje SMOTE. Pomaže kod visokodimenzionalnih podataka (libras_move, d=90).

---

### 2.10. Random-SMOTE (Dong 2011)

**Osnovna ideja:** Umjesto interpolacije prema susjedu, potpuno nasumičan smjer i udaljenost — maksimalna raznolikost.

**Kako funkcionira:** Za svaki manjinski primjer nasumično se bira smjer (jedinični vektor) i udaljenost. Nema veze sa susjedima.

**Zaključak:** Rank #5 globalno — iznenađujuće visoko! Najbolji na HIGH IR datasetovima (+0.011 vs SMOTE). Jednostavan, a efektivan.

---

### 2.11. PolynomFit-SMOTE (Gazzah 2008)

**Osnovna ideja:** Linearna interpolacija (2 točke) je prejednostavna — polinom kroz više točaka bolje prati nelinearne distribucije.

**Kako funkcionira:** Koristi polinomnu interpolaciju kroz više susjeda (degree=2 ili 3) umjesto linearnog segmenta između dvije točke.

**Zaključak:** Rank #1 globalno! Iako malo citiran u literaturi (~40 citata), u našim eksperimentima se pokazao kao najbolji za F1 (0.7508). Ovo je neočekivan i zanimljiv nalaz.

---

### 2.12. WGAN — Wasserstein GAN (u razvoju)

**Osnovna ideja:** Koristiti generativnu suprotstavljenu mrežu za stvaranje visokokvalitetnih sintetičkih primjera manjinske klase, umjesto jednostavne interpolacije.

**Kako funkcionira:** WGAN se sastoji od generatora (stvara sintetičke uzorke) i kritičara (procjenjuje koliko su uvjerljivi). Wasserstein udaljenost pruža stabilnije treniranje od klasičnog GAN-a. Generator uči stvarnu distribuciju manjinske klase i generira nove uzorke iz nje.

**Izazovi u odnosu na SMOTE:**
1. Zahtijeva **PyTorch** i GPU za efikasno treniranje (ili dugo CPU vrijeme)
2. **Mala količina podataka** — manjinska klasa često ima premalo uzoraka za treniranje GAN-a (<100)
3. **Mode collapse** — generator može proizvoditi uvijek iste uzorke
4. **Overfitting** na malom broju uzoraka — GAN nauči kopirati originale umjesto generalizirati
5. **Nestabilno treniranje** — zahtijeva pažljivo podešavanje hiperparametara
6. Računski **10-100× sporije** od SMOTE-a

**Status:** Implementacija u `smote_variants/gan.py`. U početnom stanju razvoja. Nije uključena u glavne eksperimente. Odluka o uključivanju u konačnu verziju rada ovisit će o vremenu.

---

## 3. Ključni rezultati

### 3.1. Globalni ranking (F1, svih 18 datasetova, svih 8 klasifikatora)

| # | Algoritam | F1 | vs Baseline |
|---|-----------|-----|-------------|
| 1 | **PolynomFit-SMOTE** | 0.7508 | +0.032 |
| 2 | **SMOTE-Tomek** | 0.7496 | +0.031 |
| 3 | SafeLevel-SMOTE | 0.7493 | +0.031 |
| 4 | **SMOTE** | 0.7493 | +0.031 |
| 5 | **Random-SMOTE** | 0.7487 | +0.030 |
| 15 | NoOversampling | 0.7187 | — |
| 18 | NearMiss-3 | 0.6568 | -0.062 |
| 19 | NearMiss-2 | 0.5778 | -0.141 |
| 20 | NearMiss-1 | 0.5732 | -0.146 |

**Ključni nalaz:** Top 5 su unutar 0.002 F1 — svi praktički identični. Ali SMOTE konzistentno poboljšava baseline (NoOversampling) za +0.031 F1. NearMiss poduzorkovanje je katastrofalno — gubitak 0.14 F1.

### 3.2. Po metrici

| Metrika | SMOTE | Baseline | Delta | Friedman p |
|---------|-------|----------|-------|-------------|
| F1 | 0.749 | 0.719 | **+0.031** | <0.001 |
| G-Mean | 0.822 | 0.681 | **+0.141** | <0.001 |
| AUC-ROC | 0.894 | 0.869 | **+0.025** | <0.001 |

**Ključni nalaz:** SMOTE najviše pomaže G-Mean-u (+0.141) jer izravno poboljšava balans između osjetljivosti i specifičnosti. F1 poboljšanje je skromnije (+0.031) jer SMOTE ponekad smanjuje preciznost.

### 3.3. Po IR grupi (F1)

| IR grupa | SMOTE rank | Tko je bolji? |
|----------|-----------|---------------|
| LOW IR (<5) | #4 | **NoOversampling je najbolji** — SMOTE ne pomaže kad su klase skoro balansirane |
| MED IR (5-15) | #4 | PolynomFit-SMOTE, SMOTE-Tomek — blago bolji |
| HIGH IR (>15) | #4 | **Random-SMOTE (+0.011)** i **PolynomFit-SMOTE (+0.004)** — mjerljivo bolji |

### 3.4. Po klasifikatoru — tko pobjeđuje SMOTE?

| Klasifikator | Bolji od SMOTE? |
|-------------|-----------------|
| RF, DT, LR, XGBoost | **SMOTE je najbolji** — nitko ga ne pobjeđuje |
| GNB | ENN (+0.033), TomekLinks (+0.023) |
| kNN | RandomOversampling (+0.010), Random-SMOTE (+0.010) |
| MLP | Svi su praktički jednaki |
| SVM | **Random-SMOTE (+0.014)** — jedini značajan dobitak |

### 3.5. Gdje varijante POMAŽU — per dataset

| Dataset | IR | SMOTE F1 | Najbolja varijanta | Delta |
|---------|-----|----------|---------------------|-------|
| yeast_me2 | 28.1 | 0.293 | **KMeans-SMOTE (0.305)** | +0.012 |
| wine_quality | 25.8 | 0.275 | **Random-SMOTE (0.300)** | +0.025 |
| us_crime | 12.3 | 0.466 | **KMeans-SMOTE (0.484)** | +0.018 |
| libras_move | 14.0 | 0.745 | **Random-SMOTE (0.755)** | +0.010 |
| optical_digits | 9.1 | 0.826 | **Random-SMOTE (0.868)** | +0.042 |
| abalone | 9.7 | 0.347 | **SMOTE-ENN (0.374)** | +0.027 |
| synth_noisy | — | 0.887 | **ENN (0.919)** | +0.032 |
| synth_high_ir | — | 0.918 | **ENN (0.966)** | +0.048 |

**Zaključak:** Na datasetovima sa **šumom** ENN/TomekLinks pobjeđuju (+0.03-0.05). Na **visokom IR** KMeans-SMOTE i Random-SMOTE vode. Na **niskom IR** SMOTE je optimalan.

### 3.6. Gdje SMOTE NE pomaže — per dataset

Na 6/18 datasetova **NoOversampling ≥ SMOTE**:
- breast_cancer, iris, wine (IR<3)
- synth_clean, synth_medium_ir, synth_low_dim

**Zaključak:** Kad su klase približno balansirane (IR<3) ili kad nema šuma, SMOTE je suvišan — može čak i smanjiti F1. Ovo je u skladu s literaturom i s tvojim završnim radom (Džoić, 2024).

---

## 4. Usporedba s tvojim završnim radom (Džoić, 2024)

Tvoj završni rad istraživao je utjecaj parametara k i q na osnovni SMOTE. Ključni nalazi koji se **potvrđuju** u diplomskom:

| Nalaz iz završnog (2024) | Potvrda u diplomskom (2026) |
|--------------------------|---------------------------|
| k=5 optimalno za većinu slučajeva | k=[3,5] dovoljno — k=7 ne donosi značajnu razliku |
| G-Mean raste s preuzorkovanjem, F1 može pasti | G-Mean +0.141, F1 samo +0.031 |
| SVM i GNB profitiraju najviše | SVM: Random-SMOTE +0.014; GNB: ENN +0.033 |
| Stablo odluke slabo profitira od SMOTE-a | DT: nitko ne pobjeđuje SMOTE, ali ni SMOTE ne pomaže puno (+0.006 vs baseline) |
| Parametri ovise o IR-u i klasifikatoru | Potvrđeno na 18 datasetova s 8 klasifikatora |

**Diplomski proširuje završni:** S parametara jednog algoritma → na usporedbu 12 različitih algoritama.

---

## 5. Što još treba napraviti

### Prioritet 1 — LaTeX rad (pisanje)
- [x] Struktura i plan poglavlja — gotovo (prva verzija)
- [ ] Poglavlje 2: Pretvoriti upute u gotov tekst (20-25 stranica)
- [ ] Poglavlje 3: Opisi svih SMOTE varijanti + tablica (10-15 str)
- [ ] Poglavlje 4: Opis implementacije (8-10 str)
- [ ] Poglavlje 5: Rezultati i diskusija — **ovo je najvažnije** (20-25 str)
- [ ] Poglavlje 6: Zaključak (2-3 str)
- [ ] Sažetak, abstract, literatura

### Prioritet 2 — Finalna analiza
- [x] Full run s 8 klasifikatora — **gotovo**
- [x] Statistička analiza (Friedman, Nemenyi, Wilcoxon) — **gotovo**
- [x] Vizualizacije (boxplot, violin, CD, heatmap, per-dataset bars) — **gotovo**
- [ ] Analiza po k-vrijednostima (3 vs 5 vs 7) iz full runa
- [ ] Odluka: hoće li WGAN ući u rad

### Prioritet 3 — Opcionalno
- [ ] WGAN: dovršiti implementaciju i evaluaciju (zahtijeva PyTorch, GPU)
- [ ] Parcijalno balansiranje (ne samo 1:1) — iz završnog rada
- [ ] Cross-validacija specifična po datasetu (npr. LOOCV za male skupove)

---

## 6. Struktura za prezentaciju mentoru (5 min)

1. **Što sam napravio** (30s): 20 metoda, 8 klasifikatora, 18 datasetova, 7 metrika, 54k redova rezultata
2. **Tri ključna nalaza** (2min):
   - SMOTE je robustan — nijedna varijanta ga ne nadmašuje univerzalno
   - Ali specifične varijante pomažu u specifičnim scenarijima (ENN za šum, KMeans-SMOTE za visok IR, Random-SMOTE za SVM)
   - NearMiss poduzorkovanje je konzistentno loše
3. **Demo** (1min): Web sučelje s 12 varijanti
4. **Status i plan** (1min): Full run gotov, analiza gotova, treba napisati tekst
5. **WGAN** (30s): U razvoju, opcionalno — izazovi s malom količinom podataka
