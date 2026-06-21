from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bot.nlp import extract_budget_value


def load_catalog(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def format_product(product: dict[str, Any]) -> str:
    features = ", ".join(product.get("features", []))
    genres = ", ".join(product.get("genres", []))
    return (
        f"🎵 {product['name']}\n"
        f"Категория: {product['category']}\n"
        f"Цена: {product['price_rub']} ₽\n"
        f"Для кого: {product['level']}\n"
        f"Жанры: {genres}\n"
        f"Плюсы: {features}\n"
        f"Почему может подойти: {product['pitch']}"
    )


def match_products(catalog: list[dict[str, Any]], profile: dict[str, Any]) -> list[dict[str, Any]]:
    scored: list[tuple[int, dict[str, Any]]] = []
    preferred_category = profile.get("preferred_category")
    preferred_genre = profile.get("preferred_genre")
    experience_level = profile.get("experience_level")
    budget = profile.get("budget")
    purpose = profile.get("purpose")

    budget_limit = extract_budget_value(str(budget)) if budget else None

    for product in catalog:
        score = 0
        if preferred_category and product["category"] == preferred_category:
            score += 6
        if preferred_genre and preferred_genre in product.get("genres", []):
            score += 4
        if experience_level and product["level"] in {experience_level, "универсальный"}:
            score += 3
        if purpose == "для домашней студии" and product["category"] == "клавишные":
            score += 4
        if purpose == "для ребенка" and product["price_rub"] <= 20000:
            score += 2
        if purpose == "в подарок" and product["price_rub"] <= 25000:
            score += 2
        if purpose == "для выступлений" and product["level"] in {"универсальный", "продвинутый"}:
            score += 2
        if budget_limit is not None and product["price_rub"] <= budget_limit:
            score += 6
            price_gap = budget_limit - product["price_rub"]
            if price_gap <= max(3000, int(budget_limit * 0.15)):
                score += 3
        elif budget_limit is not None:
            overshoot = product["price_rub"] - budget_limit
            if overshoot <= max(5000, int(budget_limit * 0.2)):
                score += 1
            else:
                score -= 6
        scored.append((score, product))

    scored.sort(
        key=lambda pair: (
            -pair[0],
            abs(pair[1]["price_rub"] - budget_limit) if budget_limit is not None else pair[1]["price_rub"],
            pair[1]["price_rub"],
        )
    )
    return [product for _, product in scored]
