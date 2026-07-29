import os
import sys
import json
import pickle
import argparse
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report
)

# Dodaj src/ na Python path kako bi model_utils bio dostupan
sys.path.insert(0, os.path.dirname(__file__))
from model_utils import clean_dataframe, build_text_features, NUM_FEATURES


# ── CLI argumenti ────────────────────────────────────────────────────────────

def parse_args():
    root = os.path.dirname(os.path.dirname(__file__))
    p = argparse.ArgumentParser(description="Treniraj klasifikator kategorija proizvoda")
    p.add_argument("--data-path",    default=os.path.join(root, "data", "products.csv"))
    p.add_argument("--model-path",   default=os.path.join(root, "models", "product_category_model.pkl"))
    p.add_argument("--metrics-path", default=os.path.join(root, "models", "metrics.json"))
    return p.parse_args()


# ── Pipeline builder ──────────────────────────────────────────────────────────

def make_pipeline(clf) -> Pipeline:
    """
    Gradi sklearn Pipeline s ColumnTransformerom koji:
    - TF-IDF vektorizira tekst naslova (bigrams, sublinear_tf)
    - Imputira i skalira numeričke featuree
    """
    text_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                   sublinear_tf=True, max_features=50_000)),
    ])

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        ("text", text_pipe, "text_features"),
        ("num",  num_pipe,  NUM_FEATURES),
    ])

    return Pipeline([
        ("prep", preprocessor),
        ("clf",  clf),
    ])


# ── Glavni tok ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.model_path),   exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)

    # 1. Učitaj i očisti
    print("Učitavam podatke …")
    df = pd.read_csv(args.data_path)
    df = clean_dataframe(df)
    df["text_features"] = df["Product Title"].apply(build_text_features)
    print(f"  {len(df):,} uzoraka | {df['Category Label'].nunique()} kategorija")

    # 2. Train/test split
    feature_cols = ["text_features"] + NUM_FEATURES
    X = df[feature_cols]
    y = df["Category Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Kandidati – isti skup preprocessinga, samo clf se mijenja
    # ComplementNB zahtijeva nenegativne vrijednosti pa koristi poseban pipeline
    from sklearn.preprocessing import MaxAbsScaler

    def make_nb_pipeline():
        text_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                       sublinear_tf=True, max_features=50_000)),
        ])
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  MaxAbsScaler()),  # čuva nenegativnost
        ])
        prep = ColumnTransformer([
            ("text", text_pipe, "text_features"),
            ("num",  num_pipe,  NUM_FEATURES),
        ])
        return Pipeline([("prep", prep), ("clf", ComplementNB(alpha=0.1))])

    candidates = {
        "LinearSVC":          make_pipeline(LinearSVC(max_iter=3000, C=1.0)),
        "LogisticRegression": make_pipeline(LogisticRegression(max_iter=1000, C=5.0,
                                                                class_weight="balanced")),
        "ComplementNB":       make_nb_pipeline(),
    }

    results = {}
    for name, pipe in candidates.items():
        print(f"Treniram: {name} …", end=" ", flush=True)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc      = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro")
        results[name] = {"accuracy": acc, "macro_f1": macro_f1, "pipeline": pipe, "preds": preds}
        print(f"accuracy={acc:.4f}  macro_f1={macro_f1:.4f}")

    # 4. Odaberi po Macro F1
    best_name = max(results, key=lambda k: results[k]["macro_f1"])
    best      = results[best_name]
    print(f"\n✓ Izabrani model: {best_name}  "
          f"(accuracy={best['accuracy']:.4f}  macro_f1={best['macro_f1']:.4f})")

    # 5. Classification report
    print("\n── Classification Report ───────────────────────────────────")
    print(classification_report(y_test, best["preds"]))

    # 6. Spremi model
    with open(args.model_path, "wb") as f:
        pickle.dump({"pipeline": best["pipeline"], "model_name": best_name}, f)
    print(f"Model spremljen: {args.model_path}")

    # 7. Spremi metrike u JSON
    metrics = {
        "selected_model": best_name,
        "accuracy":       round(best["accuracy"], 4),
        "macro_f1":       round(best["macro_f1"], 4),
        "weighted_f1":    round(f1_score(y_test, best["preds"], average="weighted"), 4),
        "all_models": {
            name: {
                "accuracy": round(v["accuracy"], 4),
                "macro_f1": round(v["macro_f1"], 4),
            }
            for name, v in results.items()
        },
    }
    with open(args.metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrike spremljene: {args.metrics_path}")


if __name__ == "__main__":
    main()
