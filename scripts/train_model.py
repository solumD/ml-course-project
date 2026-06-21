from __future__ import annotations

import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from bot.intent_model import IntentClassifier


def main() -> None:
    classifier = IntentClassifier.train_from_dataset(BASE_DIR / "data" / "intents_dataset.json")
    classifier.save(BASE_DIR / "models" / "intent_classifier.pkl")
    print("Model trained and saved.")


if __name__ == "__main__":
    main()
