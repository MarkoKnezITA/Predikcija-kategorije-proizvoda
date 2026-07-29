"""
train_model.py
--------------
Trains a product category classifier on products.csv and saves the best model
to models/product_classifier.pkl.

Run:
    python train_model.py

Requirements:
    pip install scikit-learn pandas numpy
"""

import os
import re
import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# 1. LOAD & CLEAN DATA
# ─────────────────────────────────────────────

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "products.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "product_classifier.pkl")


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Normalise messy category values to canonical labels
    label_map = {
        "fridge": "Fridge Freezers",
        "CPU": "CPUs",
        "Mobile Phone": "Mobile Phones",
    }
    df["Category Label"] = df["Category Label"].str.strip().map(
        lambda x: label_map.get(x, x) if isinstance(x, str) else x
    )

    # Drop rows with missing title or category
    df = df.dropna(subset=["Product Title", "Category Label"])
    df["Product Title"] = df["Product Title"].str.strip()
    df = df[df["Product Title"] != ""]

    return df


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def build_feature_text(df: pd.DataFrame) -> pd.Series:
    """
    Combine Product Title with hand-crafted signals:
    - Normalised lowercase title (base signal)
    - Extracted numbers (model numbers, capacity, screen size etc.)
    - All-caps tokens repeated (brand acronyms: USB, LED, LCD, GB, MP …)
    - Word count and title length appended as pseudo-tokens
    """
    def engineer(title: str) -> str:
        title = str(title).lower()

        # Extract numeric tokens and repeat them (capacity / model numbers are
        # strong category signals, e.g. "128gb", "55 inch")
        numbers = re.findall(r"\d+(?:\.\d+)?", title)
        num_tokens = " ".join(numbers) * 2  # double weight

        # Original (lowercased)
        words = title.split()
        word_count = len(words)
        max_word_len = max((len(w) for w in words), default=0)

        # Presence flags as pseudo-tokens
        flags = []
        if word_count > 5:
            flags.append("longtitle")
        if max_word_len > 10:
            flags.append("longword")
        if re.search(r"\d", title):
            flags.append("hasnum")

        return f"{title} {num_tokens} {' '.join(flags)}"

    return df["Product Title"].apply(engineer)


# ─────────────────────────────────────────────
# 3. TRAIN & COMPARE MODELS
# ─────────────────────────────────────────────

def train_and_evaluate():
    print("Loading data …")
    df = load_and_clean(DATA_PATH)
    print(f"  {len(df):,} samples | {df['Category Label'].nunique()} categories")

    X = build_feature_text(df)
    y = df["Category Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Candidate pipelines ──────────────────
    candidates = {
        "Logistic Regression (TF-IDF)": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, C=5.0, class_weight="balanced")),
        ]),
        "Naive Bayes (TF-IDF)": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", MultinomialNB(alpha=0.1)),
        ]),
        "Random Forest (TF-IDF)": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=15_000)),
            ("clf", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42,
                                           class_weight="balanced")),
        ]),
    }

    results = {}
    for name, pipe in candidates.items():
        print(f"\nTraining: {name}")
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = (acc, pipe)
        print(f"  Test accuracy: {acc:.4f}")

    # ── Pick best ────────────────────────────
    best_name = max(results, key=lambda k: results[k][0])
    best_acc, best_pipe = results[best_name]
    print(f"\n✓ Best model: {best_name}  (accuracy={best_acc:.4f})")

    # ── Full evaluation of best ──────────────
    preds_best = best_pipe.predict(X_test)
    print("\n── Classification Report ──────────────────")
    print(classification_report(y_test, preds_best))

    # ── Save ─────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"pipeline": best_pipe, "model_name": best_name}, f)
    print(f"\nModel saved to: {MODEL_PATH}")

    return best_pipe


if __name__ == "__main__":
    train_and_evaluate()
