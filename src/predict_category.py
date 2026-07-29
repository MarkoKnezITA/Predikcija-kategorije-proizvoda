"""
predict_category.py
-------------------
Interaktivni alat za predikciju kategorije proizvoda.

Upotreba:
    python src/predict_category.py
    python src/predict_category.py "Samsung Galaxy A52 128GB"

Model mora biti prethodno treniran:
    python src/train_model.py
"""

import os
import sys
import pickle
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from model_utils import build_text_features, NUM_FEATURES

ROOT       = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(ROOT, "models", "product_category_model.pkl")

DEMO_CASES = [
    ("iphone 7 32gb gold,4,3,Apple iPhone 7 32GB", "Mobile Phones"),
    ("olympus e m10 mark iii geh use silber",       "Digital Cameras"),
    ("kenwood k20mss15 solo",                        "Microwaves"),
    ("bosch wap28390gb 8kg 1400 spin",               "Washing Machines"),
    ("bosch serie 4 kgv39vl31g",                     "Fridge Freezers"),
    ("smeg sbs8004po",                               "Fridge Freezers"),
]


def load_model():
    if not os.path.exists(MODEL_PATH):
        print("⚠  Model nije pronađen. Pokrenite: python src/train_model.py")
        sys.exit(1)
    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)
    return obj["pipeline"], obj.get("model_name", "nepoznat")


def make_row(title: str) -> pd.DataFrame:
    """Kreira DataFrame red s tekstualnim i numeričkim featureima za jedan naslov."""
    row = {"text_features": build_text_features(title)}
    for col in NUM_FEATURES:
        row[col] = 0.0
    return pd.DataFrame([row])


def predict(title: str, pipeline) -> tuple[str, float]:
    row   = make_row(title)
    pred  = pipeline.predict(row)[0]
    # Confidence samo ako model podržava predict_proba
    try:
        proba = pipeline.predict_proba(row)[0].max()
    except AttributeError:
        proba = 1.0  # LinearSVC nema predict_proba
    return pred, proba


def run_demo(pipeline):
    print("\n── Demo test (primjeri iz zadatka) ─────────────────────────")
    correct = 0
    for title, expected in DEMO_CASES:
        predicted, conf = predict(title, pipeline)
        ok = "✓" if predicted == expected else "✗"
        if predicted == expected:
            correct += 1
        conf_str = f"{conf:.0%}" if conf < 1.0 else "—"
        print(f"  {ok}  {title[:52]:<53} → {predicted:<20} ({conf_str})")
    print(f"\nDemo točnost: {correct}/{len(DEMO_CASES)}")


def interactive_loop(pipeline, model_name):
    print(f"\n══════════════════════════════════════════")
    print(f"  Klasifikator kategorija proizvoda")
    print(f"  Model: {model_name}")
    print(f"══════════════════════════════════════════")
    print("Unesite naziv proizvoda (ili 'exit' za izlaz):\n")
    while True:
        try:
            title = input("  Naziv > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nDoviđenja!")
            break
        if title.lower() in ("exit", "quit", "q", ""):
            print("Doviđenja!")
            break
        category, conf = predict(title, pipeline)
        print(f"  → Predviđena kategorija: {category}")
        if conf < 1.0:
            print(f"  → Pouzdanost:            {conf:.1%}\n")
        else:
            print()


if __name__ == "__main__":
    pipeline, model_name = load_model()
    if len(sys.argv) > 1:
        title    = " ".join(sys.argv[1:])
        cat, conf = predict(title, pipeline)
        print(f"Predviđena kategorija: {cat}")
        if conf < 1.0:
            print(f"Pouzdanost: {conf:.1%}")
    else:
        run_demo(pipeline)
        interactive_loop(pipeline, model_name)
