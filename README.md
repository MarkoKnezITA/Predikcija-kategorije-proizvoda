# Product Category Classifier 

Automatska klasifikacija kategorije proizvoda na osnovu naziva : ML projekat za IMLP6 Task 03.

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

**Test accuracy: ~97.4%** (Logistic Regression + TF-IDF bigrams, n=35 096)

---

## Struktura projekta

```
product-classifier/
├── data/
│   └── products.csv              # Dataset (35 000+ proizvoda)
├── models/
│   └── product_classifier.pkl    # Sačuvani model (generiše train_model.py)
├── notebooks/
│   └── product_classifier_analysis.ipynb  # EDA + feature engineering + evaluacija
├── train_model.py                # Treniranje i čuvanje modela
├── predict_category.py           # Interaktivno testiranje
└── README.md
```

---

## Pokretanje

### 1. Instaliraj zavisnosti

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

### 2. Treniraj model

```bash
python train_model.py
```

Skripta će:
- učitati i očistiti `data/products.csv`
- primeniti feature engineering
- trenirati i uporediti 3 modela (Logistic Regression, Naive Bayes, Random Forest)
- sačuvati best model u `models/product_classifier.pkl`
- ispisati classification report

### 3. Testiraj model

**Interaktivni mod:**
```bash
python predict_category.py
```

**Direktno iz komandne linije:**
```bash
python predict_category.py "Samsung Galaxy A52 128GB"
# Kategorija: Mobile Phones  (pouzdanost: 99%)
```

### 4. Istraži analizu

Otvori `notebooks/product_classifier_analysis.ipynb` u Jupyter-u ili Google Colab-u za kompletan prikaz svih koraka.

---

## Metodologija

### Priprema podataka
- Normalizacija neujednačenih labela (`fridge` → `Fridge Freezers`, `CPU` → `CPUs`, itd.)
- Uklanjanje redova sa praznim naslovom ili kategorijom (~215 redova, <1%)

### Feature engineering
Svaki naslov se transformiše u obogaćeni tekst:
- **Lowercase naslov** — osnova
- **Numerički tokeni × 2** — kapacitet (128GB), težina (8kg), godina (2024), model broj — jaki signali kategorije
- **Pseudo-tokeni** — `longtitle`, `longword`, `hasnum` — enkodiraju strukturu naslova

### Modeli
| Model | Test Accuracy | Komentar |
|---|---|---|
| **Logistic Regression** | **97.39%** | ✅ Izabran — najboljii balans između preciznosti, brzine i interpretabilnosti |
| Naive Bayes | 97.26% | Brz, skoro jednako dobar |
| Random Forest | 96.00% | Sporiji, nešto slabiji na kratkim naslovima |

### Slabosti modela
- Fridge Freezers vs Fridges — kratki model kodovi (npr. `smeg sbs8004po`) ne nose dovoljno informacija; precision ~91-95%
- Rešenje: dodati eksterni rečnik brendova po kategorijama

---

## Autor

Marko Knežević
