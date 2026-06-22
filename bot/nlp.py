from __future__ import annotations

import re
from functools import lru_cache

import nltk

try:
    import pymorphy3
except ImportError:  # pragma: no cover
    pymorphy3 = None


ALLOWED_RE = re.compile(r"[^а-яё0-9a-z\s\-]", re.IGNORECASE)
SPACES_RE = re.compile(r"\s+")

GENRE_PATTERNS = {
    "рок": {
        "рок", "рока", "року", "роком", "рок-н-ролл", "рок н ролл", "панк", "панк рок", "альтернатива",
        "альтернативный рок", "метал", "металл", "хард рок", "инди рок",
    },
    "джаз": {
        "джаз", "джаза", "джазу", "джазом", "блюз", "блюза", "блюзу", "блюзом", "соул", "свинг",
    },
    "классика": {
        "классика", "классическую", "классическую музыку", "классика музыка", "академическая музыка", "фортепианная музыка",
    },
    "электроника": {
        "электроника", "электронная музыка", "электронку", "edm", "хаус", "техно", "транс", "синтвейв", "drum and bass", "dnb",
    },
    "поп": {
        "поп", "попса", "поп музыка", "поп-музыка", "популярная музыка", "эстрада",
    },
    "инди": {
        "инди", "инди музыка", "indie", "инди поп", "инди фолк", "акустика", "акустическую музыку",
    },
    "хип хоп": {
        "хип хоп", "хип-хоп", "рэп", "рэпчик", "битмейкинг", "битмейкингом", "trap", "трэп", "hip hop",
    },
}

CATEGORY_PATTERNS = {
    "гитара": {"гитара", "гитару", "гитаре", "акустическая гитара", "акустику", "электрогитара", "электрогитару"},
    "клавишные": {"пианино", "фортепиано", "клавиши", "клавишные", "синтезатор", "миди клавиатура", "миди-клавиатура"},
    "укулеле": {"укулеле", "укулель", "маленькая гитара"},
}

PURPOSE_PATTERNS = {
    "для себя": {"для себя", "для души", "для удовольствия", "для хобби", "для расслабления"},
    "в подарок": {"в подарок", "подарок", "подарить", "на подарок"},
    "для ребенка": {"для ребенка", "для детей", "для сына", "для дочки", "для девочки", "для мальчика", "ребенку", "ребёнку"},
    "для домашней студии": {"для домашней студии", "для студии", "для записи", "для битов", "писать биты", "делать биты", "для продакшна", "для аранжировок"},
    "для выступлений": {"для выступлений", "для сцены", "для концертов", "для группы", "играть в группе"},
    "для обучения": {"для обучения", "для учебы", "для учебы дома", "учиться", "обучение", "для занятий"},
}

SPELLCHECK_VOCABULARY = {
    "привет", "здравствуй", "пока", "спасибо", "погода", "музыка", "музыку", "слушаю", "люблю", "интересно",
    "рок", "джаз", "классика", "электроника", "поп", "инди", "хип", "хоп", "рэп", "блюз",
    "гитара", "гитару", "гитаре", "клавиши", "клавишные", "пианино", "фортепиано", "синтезатор",
    "укулеле", "инструмент", "инструмента", "подобрать", "выбрать", "купить", "вариант", "другой",
    "новичок", "новичком", "опыт", "играю", "бюджет", "тысяч", "дорого", "дешевле", "получше",
    "ребенка", "ребенку", "подарок", "студии", "студия", "обучение", "выступлений", "оформляем", "беру",
}

COMMON_TYPO_MAP = {
    "превет": "привет",
    "преветик": "приветик",
    "привте": "привет",
    "здраствуй": "здравствуй",
    "музка": "музыка",
    "музику": "музыку",
    "музон": "музыка",
    "гетара": "гитара",
    "гитарра": "гитара",
    "гитра": "гитара",
    "укулеля": "укулеле",
    "синтезтар": "синтезатор",
    "синтезаторр": "синтезатор",
    "клавши": "клавиши",
    "пианино": "пианино",
    "роцк": "рок",
    "джэаз": "джаз",
    "электроникаа": "электроника",
    "бютжет": "бюджет",
    "бюжет": "бюджет",
    "дорага": "дорого",
    "падешевле": "подешевле",
    "офармляем": "оформляем",
    "оформлям": "оформляем",
    "пасиба": "спасибо",
    "спасиба": "спасибо",
    "пж": "пожалуйста",
    "пжлст": "пожалуйста",
    "щас": "сейчас",
    "ща": "сейчас",
}

POSITIVE_WORDS = {
    "люблю", "обожаю", "нравится", "класс", "круто", "супер", "отлично", "хорошо", "интересно", "подходит",
    "беру", "оформляем", "хочу", "здорово", "прекрасно", "огонь",
}

POSITIVE_SMALLTALK_PHRASES = {
    "хорошая погода",
    "погода хорошая",
    "сегодня хорошая погода",
    "отличная погода",
    "классная погода",
    "сегодня классно",
    "сегодня хорошо",
    "тепло сегодня",
    "сегодня тепло",
    "солнечно",
    "сегодня солнечно",
    "да хорошая",
    "ага хорошая",
    "угу хорошая",
    "да классная",
    "да отличная",
}

NEGATIVE_WORDS = {
    "ненавижу", "плохо", "ужасно", "не нравится", "дорого", "дороговато", "не хочу", "мимо", "нет", "неа",
    "плохой", "отстой", "грустно", "скучно", "не подходит", "не то",
}


@lru_cache(maxsize=1)
def get_morph():
    if pymorphy3 is None:
        return None
    return pymorphy3.MorphAnalyzer()


def clear_text(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = ALLOWED_RE.sub(" ", text)
    text = SPACES_RE.sub(" ", text).strip()
    return text


def lemmatize_text(text: str) -> str:
    cleaned = clear_text(text)
    if not cleaned:
        return ""
    morph = get_morph()
    if morph is None:
        return cleaned
    lemmas = []
    for token in cleaned.split():
        parsed = morph.parse(token)
        lemmas.append(parsed[0].normal_form if parsed else token)
    return " ".join(lemmas)


def correct_typos(text: str) -> str:
    cleaned = clear_text(text)
    if not cleaned:
        return ""

    corrected_tokens: list[str] = []
    for token in cleaned.split():
        if token in COMMON_TYPO_MAP:
            corrected_tokens.append(COMMON_TYPO_MAP[token])
            continue
        if len(token) <= 2 or token.isdigit() or token in SPELLCHECK_VOCABULARY:
            corrected_tokens.append(token)
            continue

        best_match = token
        best_distance = None
        for candidate in SPELLCHECK_VOCABULARY:
            if abs(len(candidate) - len(token)) > 2:
                continue
            distance = nltk.edit_distance(token, candidate)
            normalized_distance = distance / max(len(candidate), len(token), 1)
            if normalized_distance <= 0.34 and (best_distance is None or normalized_distance < best_distance):
                best_match = candidate
                best_distance = normalized_distance
        corrected_tokens.append(best_match)
    return " ".join(corrected_tokens)


def analyze_sentiment(text: str) -> float:
    cleaned = clear_text(text)
    if not cleaned:
        return 0.0

    score = 0.0
    for phrase in NEGATIVE_WORDS:
        if phrase in cleaned:
            score -= 1.0
    for phrase in POSITIVE_WORDS:
        if phrase in cleaned:
            score += 1.0

    tokens = cleaned.split()
    if not tokens:
        return 0.0
    normalized = score / max(1, min(len(tokens), 5))
    return max(-1.0, min(1.0, normalized))


def is_short_confirmation(text: str) -> bool:
    return clear_text(text) in {
        "да", "ага", "угу", "давай", "конечно", "хочу", "ок", "окей", "ладно", "можно", "беру", "интересно"
    }


def is_short_rejection(text: str) -> bool:
    return clear_text(text) in {"нет", "неа", "не хочу", "не надо", "не", "мимо", "вряд ли"}


def is_affirmative_reply(text: str) -> bool:
    cleaned = clear_text(text)
    return cleaned in {
        "да", "ага", "угу", "давай", "конечно", "хочу", "ок", "окей", "ладно", "можно", "интересно",
        "почему бы и нет", "в целом да", "думаю да",
    }


def is_negative_reply(text: str) -> bool:
    cleaned = clear_text(text)
    return cleaned in {
        "нет", "неа", "не хочу", "не надо", "не", "мимо", "вряд ли", "скорее нет", "не думаю", "не особо",
    }


def is_more_request(text: str) -> bool:
    return clear_text(text) in {"еще", "ещё", "дальше", "покажи еще", "еще вариант", "другой", "другую", "следующий", "что еще", "еще что нибудь"}


def is_preference_statement(text: str) -> bool:
    cleaned = clear_text(text)
    return any(
        phrase in cleaned
        for phrase in {
            "мне нравится",
            "нравится",
            "люблю",
            "мне ближе",
            "мне подойдет",
            "мне подойдет",
            "хочу такой",
            "интересует",
        }
    )


def is_current_option_choice(text: str) -> bool:
    cleaned = clear_text(text)
    direct_phrases = {
        "давай этот",
        "беру этот",
        "этот",
        "этот вариант",
        "выбираю этот",
        "хочу этот",
        "да этот",
        "да выбираю его",
        "да беру его",
        "да этот хочу",
        "выбираю его",
        "беру его",
        "этот хочу",
        "его беру",
        "его выбираю",
    }
    return cleaned in direct_phrases or any(
        phrase in cleaned
        for phrase in {
            "давай этот",
            "беру этот",
            "этот вариант",
            "выбираю этот",
            "хочу этот",
            "выбираю его",
            "беру его",
            "этот хочу",
            "его беру",
            "его выбираю",
        }
    )


def is_order_confirmation(text: str) -> bool:
    cleaned = clear_text(text)
    confirmation_phrases = {
        "беру",
        "беру этот",
        "давай этот",
        "этот",
        "этот вариант",
        "подходит беру",
        "мне подходит",
        "мне подходит этот",
        "оформляем",
        "да оформляем",
        "хочу этот",
        "выбираю этот",
        "остановимся на этом",
        "давайте оформим",
        "создавай заказ",
        "оформи заказ",
        "да выбираю его",
        "да беру его",
        "да этот хочу",
        "выбираю его",
        "беру его",
        "выбираю его да",
    }
    return cleaned in confirmation_phrases or any(
        phrase in cleaned
        for phrase in {
            "беру",
            "оформ",
            "подходит",
            "выбираю этот",
            "хочу этот",
            "заказ",
            "выбираю его",
            "беру его",
            "его хочу",
        }
    )


def is_stop_request(text: str) -> bool:
    cleaned = clear_text(text)
    stop_phrases = {
        "стоп",
        "хватит",
        "не надо",
        "не хочу",
        "не интересно",
        "давай потом",
        "потом",
        "сменим тему",
        "не предлагай",
        "не предлагать",
        "не нужно",
    }
    return cleaned in stop_phrases


def is_smalltalk_request(text: str) -> bool:
    cleaned = clear_text(text)
    smalltalk_phrases = {
        "общение",
        "пообщаться",
        "просто общаться",
        "давай общаться",
        "просто поговорить",
        "поговорить",
        "поболтать",
        "просто поболтать",
    }
    return cleaned in smalltalk_phrases


def is_counter_question(text: str) -> bool:
    cleaned = clear_text(text)
    if not cleaned:
        return False
    exact_phrases = {
        "а ты",
        "а тебе",
        "а у тебя",
        "а сам",
        "а как у тебя",
        "а как ты",
        "а тебе как",
        "а ты как",
        "сам как",
        "ты сам как",
    }
    if cleaned in exact_phrases:
        return True

    embedded_markers = {
        "а ты",
        "а тебе",
        "а у тебя",
        "как у тебя",
        "как ты",
        "а сам",
        "ты сам как",
        "сам как",
        "а как у тебя",
        "а как ты",
        "а тебе как",
        "а ты как",
        "а ты как думаешь",
        "а ты что любишь",
        "а тебе что ближе",
        "а ты что слушаешь",
    }
    if any(marker in cleaned for marker in embedded_markers):
        return True

    tokens = cleaned.split()
    if len(tokens) >= 2:
        trailing_pairs = {
            ("а", "тебе"),
            ("а", "ты"),
            ("сам", "как"),
        }
        if tuple(tokens[-2:]) in trailing_pairs:
            return True

    trailing_triplets = {
        ("как", "у", "тебя"),
        ("а", "у", "тебя"),
        ("а", "как", "ты"),
        ("а", "как", "у"),
        ("а", "ты", "как"),
        ("а", "тебе", "как"),
        ("ты", "сам", "как"),
    }
    if len(tokens) >= 3 and tuple(tokens[-3:]) in trailing_triplets:
        return True

    return False


def is_weather_related(text: str) -> bool:
    cleaned = clear_text(text)
    if not cleaned:
        return False
    weather_markers = {
        "погод",
        "дожд",
        "солнеч",
        "жарк",
        "тепл",
        "холод",
        "ветер",
        "пасмур",
        "облач",
        "мороз",
    }
    return any(marker in cleaned for marker in weather_markers)


def is_positive_smalltalk(text: str) -> bool:
    cleaned = clear_text(text)
    if not cleaned:
        return False
    if cleaned in POSITIVE_SMALLTALK_PHRASES:
        return True
    if cleaned in {"приятная", "хорошая", "отличная", "классная", "неплохая", "солнечная", "теплая", "тёплая"}:
        return True
    return is_weather_related(cleaned) and any(
        phrase in cleaned for phrase in {"хорош", "отлич", "класс", "супер", "приятн", "тепл", "солнеч"}
    )


def is_selection_request(text: str) -> bool:
    cleaned = clear_text(text)
    selection_phrases = {
        "выбрать",
        "что выбрать",
        "давай выберем",
        "подобрать",
        "помоги выбрать",
        "хочу выбрать",
    }
    return cleaned in selection_phrases


def is_bot_about_request(text: str) -> bool:
    cleaned = clear_text(text)
    return cleaned in {
        "расскажи о себе",
        "кто ты",
        "что ты умеешь",
        "чем занимаешься",
        "что ты любишь",
    }


def is_goodbye_message(text: str) -> bool:
    cleaned = clear_text(text)
    if not cleaned:
        return False
    direct_phrases = {
        "пока",
        "пока пока",
        "до свидания",
        "до встречи",
        "бывай",
        "до связи",
        "хорошо пока",
        "ладно пока",
        "ну пока",
    }
    if cleaned in direct_phrases:
        return True
    goodbye_tokens = {"пока", "бывай"}
    words = cleaned.split()
    contextual_goodbye_prefixes = {"хорошо", "ладно", "ну", "все", "всё", "давай"}
    if (
        goodbye_tokens.intersection(words)
        and len(words) <= 3
        and (
            len(words) == 1
            or words[0] in contextual_goodbye_prefixes
            or any(phrase in cleaned for phrase in {"до свидания", "до встречи", "до связи"})
        )
        and not any(phrase in cleaned for phrase in {"пока выбираю", "пока смотрю", "пока думаю"})
    ):
        return True
    return "до свидания" in cleaned or "до встречи" in cleaned or "до связи" in cleaned


def is_interjection(text: str) -> bool:
    return clear_text(text) in {
        "ага",
        "угу",
        "мм",
        "хм",
        "ясно",
        "понятно",
        "точно",
        "ну да",
        "ладно",
        "окей",
        "ок",
    }


def detect_genre(text: str) -> str | None:
    cleaned = clear_text(text)
    lemma_text = lemmatize_text(text)
    haystacks = {cleaned, lemma_text}
    for genre, patterns in GENRE_PATTERNS.items():
        if any(pattern in haystack for haystack in haystacks for pattern in patterns):
            return genre
    return None


def detect_category(text: str) -> str | None:
    cleaned = clear_text(text)
    lemma_text = lemmatize_text(text)
    haystacks = {cleaned, lemma_text}
    for category, patterns in CATEGORY_PATTERNS.items():
        if any(pattern in haystack for haystack in haystacks for pattern in patterns):
            return category
    return None


def detect_purpose(text: str) -> str | None:
    cleaned = clear_text(text)
    lemma_text = lemmatize_text(text)
    haystacks = {cleaned, lemma_text}
    for purpose, patterns in PURPOSE_PATTERNS.items():
        if any(pattern in haystack for haystack in haystacks for pattern in patterns):
            return purpose
    return None


def detect_experience_level(text: str) -> str | None:
    cleaned = clear_text(text)
    if any(phrase in cleaned for phrase in {"играю", "играю давно", "есть опыт", "умею играть", "занимаюсь"}):
        return "продвинутый"
    if any(phrase in cleaned for phrase in {"новичок", "с нуля", "только начинаю", "учусь", "присматриваюсь", "выбираю первый"}):
        return "новичок"
    return None


def extract_budget_value(text: str) -> int | None:
    cleaned = clear_text(text)
    if not cleaned:
        return None

    compact = cleaned.replace(" ", "")

    range_match = re.search(r"(\d{1,3})(?:к|k)\s*[-–]\s*(\d{1,3})(?:к|k)", cleaned)
    if range_match:
        left = int(range_match.group(1)) * 1000
        right = int(range_match.group(2)) * 1000
        return max(left, right)

    thousand_match = re.search(r"(\d{1,3})(?:к|k)\b", compact)
    if thousand_match:
        return int(thousand_match.group(1)) * 1000

    numbers = [int(match) for match in re.findall(r"\d+", cleaned)]
    if not numbers:
        return None

    max_number = max(numbers)
    has_thousand_word = any(word in cleaned for word in {"тыс", "тысяч", "тысячи"})
    if max_number < 1000 and has_thousand_word:
        return max_number * 1000
    if max_number < 1000 and any(marker in cleaned for marker in {"до", "около", "примерно", "не дороже", "не больше"}):
        return max_number * 1000
    return max_number
