"""
model_utils.py
--------------
Dijeljene helper funkcije za pripremu podataka, normalizaciju labela
i feature engineering. Koriste ih i train_model.py i predict_category.py.
"""

import re
import pandas as pd
import numpy as np


# ── Normalizacija kategorija ─────────────────────────────────────────────────

LABEL_MAP = {
    "fridge":       "Fridges",
    "Fridge":       "Fridges",
    "CPU":          "CPUs",
    "Mobile Phone": "Mobile Phones",
}


def normalize_labels(series: pd.Series) -> pd.Series:
    """Normalizira neujednačene labele kategorija na kanonske vrijednosti."""
    return series.str.strip().map(lambda x: LABEL_MAP.get(x, x) if isinstance(x, str) else x)


# ── Čišćenje podataka ────────────────────────────────────────────────────────

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Čisti raw DataFrame:
    - Strip column names
    - Normalizira Category Label
    - Uklanja redove bez naslova ili kategorije
    - Parsira datume i kreira date featuree
    - Konvertira numeričke kolone
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    df["Category Label"] = normalize_labels(df["Category Label"])
    df["Product Title"]  = df["Product Title"].str.strip()

    df = df.dropna(subset=["Product Title", "Category Label"])
    df = df[df["Product Title"] != ""]

    # Numeričke kolone
    df["Number_of_Views"] = pd.to_numeric(df["Number_of_Views"], errors="coerce").fillna(0)
    df["Merchant Rating"] = pd.to_numeric(df["Merchant Rating"], errors="coerce").fillna(0)

    # Date features
    date_col = " Listing Date  " if " Listing Date  " in df.columns else "Listing Date"
    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        df["ListingYear"]      = dates.dt.year.fillna(0).astype(int)
        df["ListingMonth"]     = dates.dt.month.fillna(0).astype(int)
        df["ListingDayOfWeek"] = dates.dt.dayofweek.fillna(0).astype(int)
    else:
        df["ListingYear"] = df["ListingMonth"] = df["ListingDayOfWeek"] = 0

    return df


# ── Feature engineering za tekst ─────────────────────────────────────────────

def build_text_features(title: str) -> str:
    """
    Obogaćuje naslov proizvoda pseudo-tokenima:
    - Lowercase naslov
    - Numerički tokeni (kapacitet, model broj) s dvostrukom težinom
    - Pseudo-tokeni: longtitle, longword, hasnum
    """
    title_low = str(title).lower()
    numbers   = re.findall(r"\d+(?:\.\d+)?", title_low)
    num_str   = " ".join(numbers) * 2

    words        = title_low.split()
    word_count   = len(words)
    max_word_len = max((len(w) for w in words), default=0)

    flags = []
    if word_count > 5:    flags.append("longtitle")
    if max_word_len > 10: flags.append("longword")
    if re.search(r"\d", title_low): flags.append("hasnum")

    return f"{title_low} {num_str} {' '.join(flags)}"


NUM_FEATURES = ["Number_of_Views", "Merchant Rating", "ListingYear", "ListingMonth", "ListingDayOfWeek"]
