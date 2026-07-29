"""
predict_category.py
-------------------
Interaktivni alat za predikciju kategorije proizvoda.

Upotreba:
    python predict_category.py
    python predict_category.py "Samsung Galaxy A52 128GB"

Model mora biti prethodno treniran (python train_model.py).
"""

import os
import re
import sys
import pickle


MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "product_classifier.pkl")


def build_feature_text(title: str) -> str:
    """Mora biti identična funkciji u train_model.py."""
    title_low = title.lower()
    numbers = re.findall(r"\d+(?:\.\d+)?", title_low)
    num_tokens = " ".join(numbers) * 2
    words = title_low.split()
    word_count = len(words)
    max_word_len = max((len(w) for w in words), default=0)
    flags = []
    if word_count > 5:
        flags.append("longtitle")
    if max_word_len > 10:
        flags.append("longword")
    if re.search(r"\d", title_low):
        flags.append("hasnum")
    return f"{title_low} {num_tokens} {' '.join(flags)}"


def load_model():
    if not os.path.exists(MODEL_PATH):
        print("⚠  Model nije pronađen. Pokrenite prvo: python train_model.py")
        sys.exit(1)
    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)
    return obj["pipeline"], obj.get("model_name", "nepoznat")


def predict(title: str, pipeline) -> str:
    feature = build_feature_text(title)
    return pipeline.predict([feature])[0]


def predict_with_confidence(title: str, pipeline) -> tuple[str, float]:
    feature = build_feature_text(title)
    proba = pipeline.predict_proba([feature])[0]
    classes = pipeline.classes_
    best_idx = proba.argmax()
    return classes[best_idx], proba[best_idx]


# ── Demo test cases (from task spec) ─────────────────────────
DEMO_CASES = [
    ("iphone 7 32gb gold,4,3,Apple iPhone 7 32GB", "Mobile Phones"),
    ("olympus e m10 mark iii geh use silber",       "Digital Cameras"),
    ("kenwood k20mss15 solo",                        "Microwaves"),
    ("bosch wap28390gb 8kg 1400 spin",               "Washing Machines"),
    ("bosch serie 4 kgv39vl31g",                     "Fridge Freezers"),
    ("smeg sbs8004po",                               "Fridge Freezers"),
]


def run_demo(pipeline):
    print("\n── Demo test (primeri iz zadatka) ─────────────────────────────")
    correct = 0
    for title, expected in DEMO_CASES:
        predicted, conf = predict_with_confidence(title, pipeline)
        ok = "✓" if predicted == expected else "✗"
        print(f"  {ok} '{title[:55]:<55}' → {predicted:<20} ({conf:.0%} conf.)")
        if predicted == expected:
            correct += 1
    print(f"\nDemo tačnost: {correct}/{len(DEMO_CASES)}")


def interactive_loop(pipeline, model_name):
    print(f"\n══════════════════════════════════════════")
    print(f"  Klasifikator kategorija proizvoda")
    print(f"  Model: {model_name}")
    print(f"══════════════════════════════════════════")
    print("Unesite naziv proizvoda (ili 'q' za izlaz):\n")
    while True:
        try:
            title = input("  Naziv > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nDoviđenja!")
            break
        if title.lower() in ("q", "quit", "exit", ""):
            print("Doviđenja!")
            break
        category, confidence = predict_with_confidence(title, pipeline)
        print(f"  → Kategorija : {category}")
        print(f"  → Pouzdanost : {confidence:.1%}\n")


if __name__ == "__main__":
    pipeline, model_name = load_model()

    if len(sys.argv) > 1:
        # Called with argument: python predict_category.py "some title"
        title = " ".join(sys.argv[1:])
        cat, conf = predict_with_confidence(title, pipeline)
        print(f"Kategorija: {cat}  (pouzdanost: {conf:.1%})")
    else:
        run_demo(pipeline)
        interactive_loop(pipeline, model_name)
