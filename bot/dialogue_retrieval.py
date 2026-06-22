from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from random import choice
import re

import nltk

from bot.nlp import clear_text


GENERIC_WORDS = {
    "и", "а", "но", "или", "это", "как", "что", "про", "для", "мне", "меня", "тебе", "тебя",
    "вообще", "кстати", "слушай", "просто", "что-то", "что", "нибудь", "сейчас",
}

WEATHER_MARKERS = {"погода", "дожд", "солнеч", "тепло", "жарко", "ветер", "холод", "пасмур"}
EMOTION_MARKERS = {"грустно", "скучно", "настроение", "устал", "одиноко", "тоскливо"}
MUSIC_MARKERS = {
    "музыка", "плейлист", "рок", "джаз", "гитара", "клавиш", "укулеле", "пианино", "синт",
    "рэп", "блюз", "электроник", "хип",
}
SALES_MARKERS = {
    "бюджет", "подбор", "подобрать", "выбрать", "инструмент", "оформляем", "вариант", "дорого", "дешевле",
}
SMALLTALK_MARKERS = {"привет", "как", "дела", "общение", "поболтать", "хобби", "день", "что нового"}


def _content_words(text: str) -> set[str]:
    return {word for word in text.split() if word and word not in GENERIC_WORDS}


def _normalize_question(text: str) -> str:
    normalized = clear_text(text)
    normalized = re.sub(r"\b(кстати|слушай|вообще|просто)\b(?:\s+\1\b)+", r"\1", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


class DialogueRetriever:
    def __init__(self, pairs: list[tuple[str, str]]):
        self.pairs = pairs
        self.index = defaultdict(list)
        self.topic_index = defaultdict(list)
        for question, answer in pairs:
            topic = self._detect_topic(question, answer)
            self.topic_index[topic].append((question, answer))
            for word in _content_words(question) or set(question.split()):
                self.index[word].append((question, answer))

    @classmethod
    def from_file(cls, path: str | Path) -> "DialogueRetriever":
        content = Path(path).read_text(encoding="utf-8")
        raw_dialogues = content.split("\n\n")
        pairs: list[tuple[str, str]] = []
        seen: set[str] = set()
        for block in raw_dialogues:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if len(lines) < 2:
                continue
            topic = None
            question_line_index = 0
            answer_line_index = 1
            if lines[0].startswith("T: "):
                topic = clear_text(lines[0].removeprefix("T: "))
                question_line_index = 1
                answer_line_index = 2
            if len(lines) <= answer_line_index:
                continue
            question = _normalize_question(lines[question_line_index].removeprefix("- ").removeprefix("Q: "))
            answer = lines[answer_line_index].removeprefix("- ").removeprefix("A: ").strip()
            if question and question not in seen:
                seen.add(question)
                pairs.append((question, answer))
        return cls(pairs)

    def _detect_topic(self, question: str, answer: str) -> str:
        combined = f"{question} {clear_text(answer)}"
        if any(marker in combined for marker in WEATHER_MARKERS):
            return "weather"
        if any(word in combined for word in EMOTION_MARKERS):
            return "emotion"
        if any(word in combined for word in MUSIC_MARKERS):
            return "music"
        if any(word in combined for word in SALES_MARKERS):
            return "sales"
        if any(word in combined for word in SMALLTALK_MARKERS):
            return "smalltalk"
        return "general"

    def search(self, replica: str) -> str | None:
        cleaned = _normalize_question(replica)
        if not cleaned:
            return None
        words = _content_words(cleaned) or set(cleaned.split())
        topic = self._detect_topic(cleaned, "")
        short_smalltalk_replies = {
            "привет": ["Привет Как у тебя настроение?", "Привет, рад тебя видеть. О чём хочешь поболтать?"],
            "хай": ["Хай Что у тебя сегодня на уме?"],
            "ага": ["Угу, понял. Продолжай, мне интересно.", "Ага, уловил. А что тебе самому ближе сейчас?"],
            "угу": ["Угу 🙂 Можешь сказать чуть подробнее?"],
            "ясно": ["Ясно. Можем продолжить, если хочешь."],
            "скучно": ["Тогда давай оживим разговор. Можем обсудить хобби, музыку или что-нибудь лёгкое."],
            "грустно": ["Понимаю. Если хочешь, можем спокойно поговорить без спешки."],
            "норм": ["Норм — уже хорошо А чем сейчас занят?"],
        }
        if len(words) < 2 and len(cleaned) < 12:
            if cleaned in short_smalltalk_replies:
                return choice(short_smalltalk_replies[cleaned])
            return None
        candidates = []
        for word in words:
            candidates.extend(self.index.get(word, []))

        topical_candidates = []
        for question, answer in self.topic_index.get(topic, []):
            question_words = _content_words(question) or set(question.split())
            if words.intersection(question_words):
                topical_candidates.append((question, answer))

        if topical_candidates:
            candidates = topical_candidates + candidates

        best_answer = None
        best_score = -1.0
        seen_questions: set[str] = set()
        for question, answer in candidates:
            if not question:
                continue
            if question in seen_questions:
                continue
            seen_questions.add(question)
            question_words = _content_words(question) or set(question.split())
            overlap = len(words.intersection(question_words))
            if overlap == 0:
                continue
            diff = abs(len(cleaned) - len(question)) / max(len(question), 1)
            if diff > 0.65:
                continue
            distance = nltk.edit_distance(cleaned, question) / max(len(question), 1)
            jaccard = overlap / max(len(words.union(question_words)), 1)
            prefix_bonus = 0.12 if cleaned.startswith(question) or question.startswith(cleaned) else 0.0
            exact_bonus = 0.35 if cleaned == question else 0.0
            topic_bonus = 0.18 if self._detect_topic(question, answer) == topic else 0.0
            score = overlap * 0.42 + jaccard * 0.28 + (1.0 - distance) * 0.20 + prefix_bonus + exact_bonus + topic_bonus
            min_overlap = 1 if len(words) <= 2 else 2
            if overlap >= min_overlap and distance < 0.45 and score > best_score:
                best_score = score
                best_answer = answer
        return best_answer
