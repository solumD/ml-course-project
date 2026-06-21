from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from bot.nlp import clear_text, correct_typos, lemmatize_text


@dataclass(slots=True)
class IntentPrediction:
    intent: str | None
    confidence: float
    normalized_text: str


class IntentClassifier:
    def __init__(self, pipeline: Pipeline, examples_by_intent: dict[str, list[str]]):
        self.pipeline = pipeline
        self.examples_by_intent = examples_by_intent

    @classmethod
    def train_from_dataset(cls, dataset_path: str | Path) -> "IntentClassifier":
        data = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
        x_text: list[str] = []
        y: list[str] = []
        examples_by_intent: dict[str, list[str]] = {}
        for intent_name, payload in data["intents"].items():
            examples_by_intent[intent_name] = []
            for example in payload["examples"]:
                normalized = lemmatize_text(correct_typos(example))
                prepared = normalized or clear_text(example)
                x_text.append(prepared)
                y.append(intent_name)
                examples_by_intent[intent_name].append(prepared)
        pipeline = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)),
                ("clf", LinearSVC()),
            ]
        )
        pipeline.fit(x_text, y)
        return cls(pipeline, examples_by_intent)

    def predict(self, text: str) -> IntentPrediction:
        normalized = lemmatize_text(correct_typos(text)) or clear_text(text)
        if not normalized:
            return IntentPrediction(intent=None, confidence=0.0, normalized_text="")

        decision = self.pipeline.decision_function([normalized])
        classes = list(self.pipeline.named_steps["clf"].classes_)

        if len(classes) == 2 and getattr(decision, "ndim", 1) == 1:
            positive_score = float(decision[0])
            negative_score = -positive_score
            score_map = {classes[0]: negative_score, classes[1]: positive_score}
        else:
            score_row = decision[0]
            score_map = {cls_name: float(score) for cls_name, score in zip(classes, score_row)}

        sorted_scores = sorted(score_map.items(), key=lambda item: item[1], reverse=True)
        best_intent, best_score = sorted_scores[0]
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else best_score - 1.0
        margin = best_score - second_score
        similarity = self._best_example_similarity(normalized, best_intent)
        confidence = max(0.0, min(1.0, 0.55 * self._sigmoid(margin) + 0.45 * similarity))

        if confidence < 0.58:
            return IntentPrediction(intent=None, confidence=confidence, normalized_text=normalized)
        return IntentPrediction(intent=best_intent, confidence=confidence, normalized_text=normalized)

    def _best_example_similarity(self, normalized_text: str, intent: str) -> float:
        examples = self.examples_by_intent.get(intent, [])
        if not examples:
            return 0.0
        best = 0.0
        for example in examples:
            max_len = max(len(example), len(normalized_text), 1)
            distance = nltk.edit_distance(normalized_text, example)
            similarity = 1.0 - (distance / max_len)
            if similarity > best:
                best = similarity
        return max(0.0, best)

    @staticmethod
    def _sigmoid(value: float) -> float:
        return value / (1.0 + abs(value)) if value != 0 else 0.0

    def save(self, model_path: str | Path) -> None:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as file:
            pickle.dump(
                {
                    "pipeline": self.pipeline,
                    "examples_by_intent": self.examples_by_intent,
                },
                file,
            )

    @classmethod
    def load(cls, model_path: str | Path) -> "IntentClassifier":
        with open(model_path, "rb") as file:
            payload = pickle.load(file)
        if isinstance(payload, dict):
            return cls(payload["pipeline"], payload.get("examples_by_intent", {}))
        return cls(payload, {})
