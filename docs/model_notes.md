# Model Notes

Kratke tehničke bilješke o pipeline-u, featureima i izlaznim fajlovima.

## Pipeline arhitektura

```
ColumnTransformer
├── text_features  → TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, max_features=50000)
└── num_features   → SimpleImputer(median) → StandardScaler
        └── Number_of_Views, Merchant Rating, ListingYear, ListingMonth, ListingDayOfWeek
↓
Classifier (LinearSVC / LogisticRegression / ComplementNB)
```

Model se bira automatski na osnovu **Macro F1 score** na test splitu (80/20, stratified).

## Feature engineering – tekst

Svaki `Product Title` se transformira funkcijom `build_text_features()`:

1. Lowercase
2. Numerički tokeni se ponavljaju × 2 (kapacitet, model broj su jaki signali)
3. Pseudo-tokeni: `longtitle` (>5 reči), `longword` (max dužina reči >10), `hasnum`

## Normalizacija labela

| Originalna labela | Kanonska labela |
|---|---|
| `fridge` | `Fridges` |
| `CPU` | `CPUs` |
| `Mobile Phone` | `Mobile Phones` |

## Izlazni fajlovi

| Fajl | Sadržaj |
|---|---|
| `models/product_category_model.pkl` | Pickle sa `{"pipeline": ..., "model_name": ...}` |
| `models/metrics.json` | Accuracy, Macro F1, Weighted F1 za sve kanditate i odabrani model |

## Poznate slabosti

- **Fridge Freezers vs Fridges** – kratki model kodovi (npr. `smeg sbs8004po`) ne nose
  dovoljno tekstualnih signala; ove dvije kategorije su najteže za razlikovati.
- LinearSVC ne daje `predict_proba`, pa `predict_category.py` ne prikazuje postotak
  pouzdanosti kada je taj model odabran.

## Moguća poboljšanja

- Brand dictionary (mapiranje brenda na kategoriju)
- Character-level n-grams za kratke model kodove
- Hyperparameter tuning (`GridSearchCV`)
- Dodavanje više title-based featurea (broj znakova, broj specijalnih znakova)
