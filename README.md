# Predikcija kategorije proizvoda 🛍️

Automatska ML klasifikacija kategorije proizvoda na osnovu naziva : IMLP6 Task 03.

## Pregled

Model prima naziv proizvoda (npr. `"Bosch WAP28390GB 8kg 1400 Spin"`) i predviđa jednu od 10 kategorija:

| Kategorija | Primjer |
|---|---|
| Mobile Phones | iPhone 7 32GB Gold |
| Digital Cameras | Olympus E-M10 Mark III |
| Microwaves | Kenwood K20MSS15 |
| Washing Machines | Bosch WAP28390GB |
| Fridge Freezers | Bosch Serie 4 KGV39VL31G |
| Fridges | Samsung RR39M7000SA |
| Freezers | Hotpoint RZSAV22P |
| Dishwashers | Bosch SMS46JI05G |
| TVs | LG 55UK6300PLB |
| CPUs | Intel Core i7-9700K |

**Test accuracy: ~98.1% | Macro F1: ~98.1%** (LinearSVC + TF-IDF bigrams + numerički featurei)

---

## Struktura projekta

```
product-classifier/
├── data/
│   └── products.csv                         # Dataset (35 000+ proizvoda)
├── docs/
│   └── model_notes.md                       # Tehničke bilješke o pipeline-u i featureima
├── models/
│   ├── product_category_model.pkl           # Sačuvani model
│   └── metrics.json                         # Metrike svih kandidata i odabranog modela
├── notebooks/
│   └── product_classifier_analysis.ipynb   # EDA + feature engineering + evaluacija
├── src/
│   ├── model_utils.py                       # Helper funkcije (čišćenje, feature engineering)
│   ├── train_model.py                       # Treniranje i čuvanje modela
│   └── predict_category.py                  # Interaktivno testiranje
├── requirements.txt
└── README.md
```

---

## Pokretanje

### 1. Instaliraj zavisnosti

```bash
pip install -r requirements.txt
```

### 2. Treniraj model

```bash
python src/train_model.py
```

Opcijski argumenti:

```bash
python src/train_model.py --data-path data/products.csv \
                           --model-path models/product_category_model.pkl \
                           --metrics-path models/metrics.json
```

Skripta će:
- učitati i očistiti `data/products.csv`
- primijeniti feature engineering (TF-IDF + numerički featurei)
- trenirati i usporediti 3 modela: LinearSVC, LogisticRegression, ComplementNB
- odabrati best model po **Macro F1 score**
- sačuvati model u `.pkl` i metrike u `metrics.json`

### 3. Testiraj model

**Interaktivni mod:**
```bash
python src/predict_category.py
```

**Direktno iz komandne linije:**
```bash
python src/predict_category.py "Samsung Galaxy A52 128GB"
# Predviđena kategorija: Mobile Phones
```

### 4. Istraži analizu

Otvori `notebooks/product_classifier_analysis.ipynb` u Jupyter-u ili Google Colab-u.

---

## Metodologija

### Čišćenje podataka
- Normalizacija neujednačenih labela (`fridge → Fridges`, `CPU → CPUs`, `Mobile Phone → Mobile Phones`)
- Uklanjanje redova s praznim naslovom ili kategorijom (~215 redova, <1%)
- Parsiranje datuma u godinu, mjesec i dan tjedna

### Feature engineering
Kombinirani `ColumnTransformer` pipeline:
- **Tekst** — TF-IDF bigrams (sublinear_tf, max 50k featurea) na obogaćenom naslovu
- **Numerički** — Number_of_Views, Merchant Rating, ListingYear, ListingMonth, ListingDayOfWeek

Naslov se obogaćuje pseudo-tokenima: numerički tokeni ×2 (kapacitet, model kod), `longtitle`, `longword`, `hasnum`.

### Poređenje modela

| Model | Test Accuracy | Macro F1 |
|---|---|---|
| **LinearSVC** ✅ | **98.08%** | **98.06%** |
| LogisticRegression | 97.83% | 97.76% |
| ComplementNB | 97.61% | 97.53% |

Model se bira automatski po Macro F1 score.

### Slabosti
- **Fridge Freezers vs Fridges** — kratki model kodovi (npr. `smeg sbs8004po`) teško razlikovati
- LinearSVC ne daje postotak pouzdanosti (nema `predict_proba`)

---

## Autor

Marko Knežević
