from __future__ import annotations

from dataclasses import asdict
import random

from bot.catalog import format_product, match_products
from bot.nlp import (
    clear_text,
    detect_category,
    detect_experience_level,
    detect_genre,
    detect_purpose,
    extract_budget_value,
    is_current_option_choice,
    is_interjection,
    is_more_request,
    is_order_confirmation,
    is_preference_statement,
    is_short_confirmation,
    is_short_rejection,
    is_stop_request,
)
from bot.state import UserProfile


MUSIC_HOOKS = [
    "Кстати, а музыка тебе вообще близка — слушаешь что-то регулярно?",
    "Раз уж заговорили об интересах: тебе ближе музыка, кино или что-то совсем другое?",
    "Иногда по хобби проще понять, что человеку действительно нравится. Музыка тебе интересна?",
]

GENRE_FOLLOWUPS = {
    "рок": "Тогда логично смотреть в сторону гитарного или универсального инструмента. Ты больше хочешь играть сам или пока просто присматриваешься?",
    "джаз": "Тогда важно, чтобы инструмент позволял чувствовать гармонию и нюансы. Ты уже играл раньше или только присматриваешься?",
    "классика": "Для такого вкуса часто хорошо заходят клавишные или акустические инструменты. Ты уже занимаешься музыкой или пока только думаешь начать?",
    "электроника": "Тогда может подойти что-то для домашней студии или клавишный формат. Ты хочешь играть руками, записывать музыку или пока просто смотришь варианты?",
    "поп": "Понятно, это даёт большой выбор. Тогда уточню: тебе ближе гитара, клавиши или что-то максимально простое для старта?",
    "инди": "Хороший ориентир. Для такого настроения часто берут гитару или укулеле. Ты хочешь начать с нуля или уже есть опыт?",
    "хип хоп": "Тогда может быть интересен студийный формат и клавишные для битов и гармоний. Ты хочешь писать музыку или просто попробовать инструмент для старта?",
}

SELECTION_INVITES = [
    "Давай подберём. Для начала быстро пойму твой вкус: музыка тебе вообще близка — слушаешь что-то регулярно?",
    "Хорошо, давай начнём с пары простых вещей. Музыка тебе в целом близка или это скорее новая тема для тебя?",
    "Супер, тогда сначала не буду гадать наугад. Скажи, музыка для тебя вообще важная часть жизни?",
]

EXPERIENCE_QUESTIONS = [
    "Класс. А у тебя уже есть опыт игры на каком-то инструменте или это будет первый?",
    "Отлично. Тогда подскажи: ты уже играл раньше на инструментах или сейчас выбираешь самый первый?",
    "Здорово. Чтобы двигаться точнее, скажи: у тебя уже есть опыт игры или ты стартуешь с нуля?",
]

EXPERIENCE_FALLBACKS = [
    "Чтобы подобрать вариант точнее, мне важно понять именно опыт игры: ты уже играл на инструментах или это будет первый?",
    "С опытом игры пока не до конца ясно. Скажи коротко: новичок ты или уже играл раньше?",
    "Давай уточним только один момент: у тебя уже есть опыт игры на инструменте или ты стартуешь с нуля?",
]

GENRE_QUESTIONS = [
    "Отлично, уже понятнее. А какой жанр тебе ближе всего: рок, поп, джаз, классика, электроника, инди или хип-хоп?",
    "Хорошо. Теперь скажи, какой жанр тебе ближе: рок, джаз, поп, классика, электроника, инди или хип-хоп?",
    "Супер. Для следующего шага нужен жанр: что тебе ближе всего по музыке?",
]

CATEGORY_QUESTIONS = [
    "Хорошо. Тогда уточню сам формат: тебе ближе гитара, клавишные или что-то компактное вроде укулеле?",
    "Окей, теперь про сам инструмент: больше тянет к гитаре, клавишным или укулеле?",
    "Понял. А по формату что тебе ближе — гитара, клавиши или что-то компактное?",
]

PURPOSE_QUESTIONS = [
    "И последний важный момент перед рекомендацией: инструмент нужен для себя, для обучения, для ребёнка, в подарок или, например, для домашней студии?",
    "Осталось понять сценарий: берёшь для себя, для подарка, для обучения, ребёнку или для домашней студии?",
    "Хорошо, и ещё один момент: инструмент нужен для каких целей?",
]

RECOMMEND_LIKE_RESPONSES = [
    "Отлично, значит этот вариант тебе откликается. Если решишь брать его, я это пойму и помогу зафиксировать выбор.",
    "Понял, этот инструмент тебе зашёл. Если готов остановиться на нём, просто скажи об этом своими словами — я продолжу оформление.",
    "Хорошо, вижу, что вариант понравился. Когда будешь готов брать, просто дай знать, и я пойду дальше.",
]

BUDGET_QUESTIONS = [
    "Чтобы не советовать что-то слишком дорогое или слабое, подскажи примерный бюджет на инструмент.",
    "Дальше осталось понять бюджет. Напиши ориентир по сумме, чтобы я не промахнулся с вариантами.",
    "Хорошо, теперь нужен примерный бюджет — так я смогу подобрать что-то реально подходящее.",
]

ALTERNATIVE_FOLLOWUPS = [
    "Если хочешь, могу предложить ещё один вариант в этом же стиле или пересобрать подборку под другой бюджет.",
    "Если этот не до конца попал, я могу показать ещё один похожий вариант или уйти в другой бюджет.",
    "Если интересно, дам альтернативу в похожем духе или подберу что-то попроще либо серьёзнее.",
]

RECOMMENDATION_BRIDGES = [
    "Если вариант тебе подходит, можем сразу зафиксировать его — не обязательно точной фразой, я пойму обычное подтверждение.",
    "Если чувствуешь, что это твой вариант, можем сразу переходить к оформлению — просто скажи, что хочешь его взять.",
    "Если инструмент откликается по звучанию и задаче, можем остановиться на нём и оформить выбор без формальных команд.",
]

DIRECT_SELECTION_OPENERS = [
    "Давай быстро и по делу подберём инструмент. Я задам только нужные вопросы.",
    "Хорошо, идём сразу в подбор без лишних кругов. Уточню только важное.",
    "Отлично, тогда включаю быстрый подбор: коротко соберём параметры и сразу перейдём к варианту.",
]


def maybe_switch_to_music(profile: UserProfile, intent: str) -> str | None:
    if (
        profile.current_stage == "smalltalk"
        and not profile.discussed_music
        and not profile.music_hook_offered
        and profile.conversation_turns >= 4
        and intent in {"hobby", "smalltalk"}
    ):
        profile.current_stage = "music_hook"
        profile.last_bot_question = "music_interest"
        available_hooks = [hook for hook in MUSIC_HOOKS if hook != profile.last_music_hook] or MUSIC_HOOKS
        chosen_hook = random.choice(available_hooks)
        profile.last_music_hook = chosen_hook
        profile.music_hook_offered = True
        return chosen_hook
    return None


def handle_sales_flow(profile: UserProfile, intent: str, text: str, catalog: list[dict]) -> str | None:
    cleaned_text = clear_text(text)
    detected_genre = detect_genre(cleaned_text)
    detected_category = detect_category(cleaned_text)
    detected_experience = detect_experience_level(cleaned_text)
    detected_purpose = detect_purpose(cleaned_text)

    if detected_purpose:
        profile.purpose = detected_purpose
    if detected_genre:
        profile.preferred_genre = detected_genre
    if detected_category:
        profile.preferred_category = detected_category
    if detected_experience:
        profile.experience_level = detected_experience

    if profile.current_stage == "smalltalk":
        if intent in {"buy_interest", "instrument_interest", "genre"} or detected_category or detected_genre or detected_purpose:
            profile.current_stage = "music_hook"
            profile.discussed_music = True
            profile.direct_selection_mode = True

            if profile.preferred_genre or profile.preferred_category or profile.purpose:
                profile.music_interest = True
                profile.current_stage = "experience"
                profile.last_bot_question = "experience"

                if profile.preferred_genre:
                    return random.choice(DIRECT_SELECTION_OPENERS) + " " + GENRE_FOLLOWUPS.get(
                        profile.preferred_genre,
                        "Здорово. А у тебя уже есть опыт игры на каком-то инструменте или ты только присматриваешься?",
                    )
                if profile.preferred_category and not profile.experience_level:
                    return random.choice(DIRECT_SELECTION_OPENERS) + " " + random.choice(EXPERIENCE_QUESTIONS)
                if profile.preferred_category and profile.experience_level and not profile.preferred_genre:
                    profile.last_bot_question = "experience"
                    return random.choice(DIRECT_SELECTION_OPENERS) + " " + random.choice(GENRE_QUESTIONS)
                return random.choice(DIRECT_SELECTION_OPENERS) + " " + random.choice(EXPERIENCE_QUESTIONS)

            profile.last_bot_question = "music_interest"
            if profile.direct_selection_mode:
                profile.music_interest = True
                profile.current_stage = "experience"
                profile.last_bot_question = "experience"
                return random.choice(DIRECT_SELECTION_OPENERS) + " " + random.choice(EXPERIENCE_QUESTIONS)
            return random.choice(SELECTION_INVITES)
        if intent == "music_yes":
            profile.music_interest = True
            profile.discussed_music = True
            profile.current_stage = "experience"
            profile.last_bot_question = "experience"
            return random.choice(EXPERIENCE_QUESTIONS)

    if profile.current_stage == "recommend" and is_stop_request(cleaned_text):
        profile.current_stage = "smalltalk"
        profile.last_bot_question = None
        return "Хорошо, не буду сейчас навязывать варианты. Если позже захочешь вернуться к выбору инструмента — просто скажи, и я спокойно помогу. Можем пока поговорить на другую тему."

    if profile.current_stage == "music_hook":
        music_affirmation = any(
            phrase in cleaned_text
            for phrase in {
                "да слушаю",
                "да музыку слушаю",
                "слушаю",
                "слушаю регулярно",
                "часто слушаю",
                "люблю музыку",
                "музыка интересна",
                "мне нравится музыка",
            }
        )
        if intent in {"music_yes", "genre", "instrument_interest"} or is_short_confirmation(cleaned_text) or detected_genre or detected_purpose or music_affirmation:
            profile.music_interest = True
            profile.discussed_music = True
            profile.current_stage = "experience"
            profile.last_bot_question = "experience"
            if detected_genre:
                return GENRE_FOLLOWUPS.get(detected_genre, "Здорово. А у тебя уже есть опыт игры на каком-то инструменте или ты только присматриваешься?")
            return random.choice(EXPERIENCE_QUESTIONS)
        if intent == "music_no" or is_short_rejection(cleaned_text):
            profile.music_interest = False
            profile.discussed_music = True
            profile.current_stage = "soft_interest"
            profile.last_bot_question = "soft_interest"
            return "Понял. А если бы выбирать что-то для расслабления или нового хобби, что звучит интереснее: гитара, клавиши или что-то компактное вроде укулеле?"
        profile.last_bot_question = "music_interest"
        return "Я пока не до конца понял ответ. Музыка тебе в целом интересна или лучше поговорить о чём-то другом?"

    if profile.current_stage in {"experience", "soft_interest"}:
        understood = False

        if detected_genre:
            understood = True

        if detected_category:
            understood = True

        if detected_purpose:
            understood = True

        if detected_experience:
            understood = True

        if intent in {"beginner", "instrument_interest"}:
            profile.experience_level = "новичок"
            understood = True
        elif intent == "advanced":
            profile.experience_level = "продвинутый"
            understood = True

        if is_short_confirmation(cleaned_text) and profile.current_stage == "soft_interest":
            understood = True
        if is_short_rejection(cleaned_text) and profile.current_stage == "soft_interest":
            return "Хорошо, без спешки. Если всё же захочешь, я могу потом предложить что-то совсем простое и недорогое для первого знакомства с музыкой."

        if intent == "genre" and detected_genre:
            understood = True

        if "присматрива" in cleaned_text or "пока смотрю" in cleaned_text or "пока выбираю" in cleaned_text:
            profile.experience_level = profile.experience_level or "новичок"
            understood = True

        if is_preference_statement(cleaned_text) and (detected_category or detected_genre or detected_purpose):
            understood = True

        if is_interjection(cleaned_text) and profile.last_bot_question == "experience" and not profile.experience_level:
            profile.last_bot_question = "experience"
            return random.choice(EXPERIENCE_FALLBACKS)

        if profile.last_bot_question == "experience" and is_short_confirmation(cleaned_text) and not profile.experience_level:
            profile.last_bot_question = "experience"
            return random.choice(EXPERIENCE_FALLBACKS)

        if not understood:
            if not profile.experience_level:
                profile.last_bot_question = "experience"
                return random.choice(EXPERIENCE_FALLBACKS)
            if not profile.preferred_genre:
                profile.last_bot_question = "experience"
                return random.choice(GENRE_QUESTIONS)
            if not profile.preferred_category:
                profile.last_bot_question = "experience"
                return random.choice(CATEGORY_QUESTIONS)
            if not profile.purpose:
                profile.last_bot_question = "experience"
                return random.choice(PURPOSE_QUESTIONS)
            profile.last_bot_question = "experience"
            return "Я почти всё понял, но лучше уточни одним сообщением недостающий момент, и я сразу продолжу подбор."

        if not profile.preferred_genre:
            profile.last_bot_question = "experience"
            return random.choice(GENRE_QUESTIONS)

        if not profile.preferred_category:
            profile.last_bot_question = "experience"
            return random.choice(CATEGORY_QUESTIONS)

        if not profile.purpose:
            profile.last_bot_question = "experience"
            return random.choice(PURPOSE_QUESTIONS)

        profile.current_stage = "budget"
        profile.last_bot_question = "budget"
        return random.choice(BUDGET_QUESTIONS)

    if profile.current_stage == "budget":
        if is_stop_request(cleaned_text):
            profile.current_stage = "smalltalk"
            profile.last_bot_question = None
            return "Без проблем, не будем сейчас углубляться в покупку. Если захочешь позже вернуться к подбору инструмента, я помогу."
        budget_value = extract_budget_value(cleaned_text)
        if budget_value is not None:
            profile.budget = budget_value
            profile.current_stage = "recommend"
            profile.last_bot_question = "recommend"
            return recommend_product(profile, catalog)
        if is_short_confirmation(cleaned_text):
            profile.last_bot_question = "budget"
            return "Тогда назови хотя бы примерную сумму: до 10000, около 20000 или выше."
        return "Можно прямо примерно: например 10000, 20000 или 30000 рублей. Так я точнее подберу вариант."

    if profile.current_stage == "recommend":
        if is_order_confirmation(cleaned_text) or is_current_option_choice(cleaned_text):
            chosen_product = _get_last_recommended_product(profile, catalog)
            chosen_name = chosen_product["name"] if chosen_product else "выбранный инструмент"
            profile.reset()
            return (
                f"Отлично, зафиксировал ваш выбор: {chosen_name}. Заказ создан, перенаправляю вас на оператора. "
                f"Если позже захотите начать новый подбор, можем начать с чистого листа."
            )
        if is_preference_statement(cleaned_text):
            chosen_product = _get_last_recommended_product(profile, catalog)
            profile.last_bot_question = "recommend"
            return random.choice(RECOMMEND_LIKE_RESPONSES)
        if intent == "buy_interest" or is_short_confirmation(cleaned_text):
            profile.last_bot_question = "recommend"
            return "Отличный выбор. Если ты уже готов взять этот инструмент, я могу сразу зафиксировать выбор и перейти дальше."
        if intent == "reject_product" or is_more_request(cleaned_text):
            return recommend_product(profile, catalog, exclude_previous=True)
        if is_short_rejection(cleaned_text):
            profile.last_bot_question = "recommend"
            return "Хорошо. Если хочешь, я могу предложить другой инструмент, а если лучше остановиться или сменить тему — тоже нормально, просто скажи."
        if any(phrase in cleaned_text for phrase in {"дешевле", "подешевле", "дороговато", "что нибудь дешевле"}):
            if profile.budget:
                profile.budget = max(3000, int(profile.budget * 0.8))
            profile.last_bot_question = "recommend"
            return recommend_product(profile, catalog, exclude_previous=True)
        if any(phrase in cleaned_text for phrase in {"получше", "что нибудь лучше", "премиум", "повыше классом"}):
            if profile.budget:
                profile.budget = int(profile.budget * 1.25)
            profile.last_bot_question = "recommend"
            return recommend_product(profile, catalog, exclude_previous=True)

    return None


def recommend_product(profile: UserProfile, catalog: list[dict], exclude_previous: bool = False) -> str:
    ranked = match_products(catalog, asdict(profile))
    if exclude_previous:
        ranked = [item for item in ranked if item["id"] not in profile.last_recommended_product_ids]
    if not ranked:
        profile.last_recommended_product_ids.clear()
        ranked = match_products(catalog, asdict(profile))
    if not ranked:
        return "Пока не вижу идеального совпадения, но могу предложить другой формат: скажи, тебе ближе гитара, клавиши или что-то компактное и простое в освоении?"
    product = ranked[0]
    profile.selected_product_id = product["id"]
    profile.last_recommended_product_ids.append(product["id"])
    profile.last_recommended_product_ids = profile.last_recommended_product_ids[-10:]
    profile.current_stage = "recommend"
    profile.last_bot_question = "recommend"
    followup = random.choice(RECOMMENDATION_BRIDGES)
    purpose_hint = ""
    if profile.purpose == "для домашней студии":
        purpose_hint = " Я учитывал, что тебе важен домашний студийный сценарий."
    elif profile.purpose == "для ребенка":
        purpose_hint = " Я держал в уме, что инструмент подбирается для ребёнка, поэтому смотрел на более дружелюбные и понятные варианты."
    elif profile.purpose == "в подарок":
        purpose_hint = " Я ориентировался на то, чтобы вариант был удачным именно как подарок."
    direct_hint = ""
    if profile.direct_selection_mode:
        direct_hint = " Я сократил путь и опирался только на ключевые параметры, которые ты уже дал."
        profile.direct_selection_mode = False
    return format_product(product) + purpose_hint + direct_hint + "\n\n" + followup


def _get_last_recommended_product(profile: UserProfile, catalog: list[dict]) -> dict | None:
    product_id = profile.selected_product_id or (profile.last_recommended_product_ids[-1] if profile.last_recommended_product_ids else None)
    if not product_id:
        return None
    for product in catalog:
        if product["id"] == product_id:
            return product
    return None
