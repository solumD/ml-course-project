from __future__ import annotations

import json
import random
from pathlib import Path

from bot.catalog import load_catalog
from bot.dialogue_retrieval import DialogueRetriever
from bot.intent_model import IntentClassifier, IntentPrediction
from bot.nlp import (
    analyze_sentiment,
    clear_text,
    is_counter_question,
    correct_typos,
    detect_category,
    detect_experience_level,
    detect_genre,
    detect_purpose,
    is_affirmative_reply,
    is_current_option_choice,
    is_bot_about_request,
    is_goodbye_message,
    is_more_request,
    is_negative_reply,
    is_positive_smalltalk,
    is_order_confirmation,
    is_preference_statement,
    is_selection_request,
    is_short_confirmation,
    is_short_rejection,
    is_smalltalk_request,
    is_weather_related,
)
from bot.sales_funnel import handle_sales_flow, maybe_switch_to_music
from bot.state import BotReply, UserProfile


SMALLTALK_FALLBACKS = [
    "Можем спокойно поболтать. Например, про музыку, фильмы, хобби или просто как проходит день.",
    "Я за живой разговор. Если хочешь, можем обсудить настроение, интересы или что-нибудь лёгкое и повседневное.",
    "Давай просто пообщаемся. Могу поддержать тему про хобби, музыку, отдых или что у тебя сейчас на уме.",
    "Можем без напряга поболтать. Если по пути всплывёт музыка или хобби — тоже легко подхвачу.",
    "Можем поговорить про то, что тебе сейчас действительно интересно. А если в разговоре зацепимся за музыку, я потом аккуратно помогу и с выбором инструмента.",
    "Я не тороплю с подбором. Давай сначала просто поймём, что тебе вообще нравится, а дальше уже можно мягко перейти к музыке и инструментам.",
]

INTERJECTION_REPLIES = [
    "Понимаю. Можешь продолжить мысль чуть подробнее — мне правда интересно.",
    "Да, бывает. Если хочешь, можем развить тему дальше.",
    "Угу, уловил. А что тебе самому сейчас интереснее всего обсудить?",
]

GREETING_WITH_QUESTION = [
    "Привет 🙂 Рад тебя видеть. Если хочешь, можем просто спокойно пообщаться.",
    "Привет! Я на связи — можем поболтать на любую лёгкую тему или обсудить интересы.",
    "Привет! Давай просто пообщаемся. Если разговор сам выйдет на музыку — я подхвачу.",
]

COUNTER_QUESTION_REPLIES = [
    "У меня всё спокойно: люблю живой разговор, музыку и темы про увлечения. А у тебя что сейчас ближе?",
    "Если коротко, мне ближе разговоры про интересы, музыку и хобби. А тебе что сегодня интереснее?",
    "Мне обычно интересны музыка, увлечения и спокойный разговор без спешки. А ты сам что бы хотел обсудить?",
]

WEATHER_COUNTER_REPLIES = [
    "У меня всё спокойно — такая погода мне тоже нравится. А тебе больше по душе солнце, прохлада или что-то дождливое и уютное?",
    "Мне тоже ближе приятная спокойная погода. А ты сам любишь скорее тепло и солнце или что-то более прохладное?",
    "Мне такая погода тоже кажется комфортной. А у тебя она обычно больше под настроение для прогулки, музыки или просто отдыха?",
]

MOOD_COUNTER_REPLIES = [
    "У меня всё спокойно, спасибо. А у тебя день сегодня больше бодрый или хочется чего-то расслабленного?",
    "Спасибо, у меня тоже всё нормально. А тебе сегодня ближе активный настрой или спокойный вечер?",
    "У меня всё хорошо. А ты сам сейчас в каком настроении — больше на общение, отдых или что-то творческое?",
]

POSITIVE_SMALLTALK_REPLIES = [
    "Согласен, настроение у такой погоды приятное. В такие дни особенно хорошо включить любимую музыку. Ты обычно что слушаешь?",
    "Да, когда погода радует, день ощущается легче. Кстати, у тебя музыка тоже часто идёт фоном в такие моменты?",
    "Понимаю, такая погода правда задаёт хороший тон. А из музыки тебе ближе что-то спокойное или более энергичное?",
    "Есть в такой погоде что-то уютное. Часто в такие моменты особенно тянет включить любимый плейлист. Что у тебя там обычно звучит?",
    "Да, такая погода легко располагает к хорошему настроению. Если хочешь, можем от этого плавно перейти к музыке — что тебе ближе по звучанию?",
    "В такие моменты обычно хорошо чувствуется, что человеку реально нравится. У тебя музыка часто рядом в течение дня или не особо?",
]

PROACTIVE_SMALLTALK_PROMPTS = [
    "Кстати, а тебе самому какие темы обычно приятнее: музыка, хобби, кино, техника или что-то совсем другое?",
    "Если не спешить с подбором, мне сначала интересно понять тебя чуть лучше. Что тебе вообще ближе как интерес?",
    "Могу просто поддержать разговор. Скажи, что тебе обычно самому интереснее обсуждать — музыку, увлечения или что-то бытовое?",
]

PROACTIVE_SMALLTALK_QUESTIONS = [
    "Какие темы тебе обычно ближе: музыка, хобби, фильмы или что-то совсем другое?",
    "А что тебе самому сейчас интереснее обсудить — музыку, увлечения или что-то повседневное?",
]


class BotEngine:
    def __init__(self, data_dir: str | Path, model_path: str | Path):
        self.data_dir = Path(data_dir)
        self.model_path = Path(model_path)
        self.intent_config = json.loads((self.data_dir / "intents_dataset.json").read_text(encoding="utf-8"))
        self.retriever = DialogueRetriever.from_file(self.data_dir / "dialogues.txt")
        self.catalog = load_catalog(self.data_dir / "products.json")
        self.model = self._load_or_train_model()

    def _load_or_train_model(self) -> IntentClassifier:
        dataset_path = self.data_dir / "intents_dataset.json"
        if self.model_path.exists() and self.model_path.stat().st_mtime >= dataset_path.stat().st_mtime:
            return IntentClassifier.load(self.model_path)
        classifier = IntentClassifier.train_from_dataset(dataset_path)
        classifier.save(self.model_path)
        return classifier

    def reply(self, text: str, profile: UserProfile) -> BotReply:
        profile.conversation_turns += 1
        raw_cleaned = clear_text(text)
        corrected_text = correct_typos(text)
        cleaned = clear_text(corrected_text)
        sentiment = analyze_sentiment(corrected_text)
        heuristic_intent = self._resolve_contextual_short_intent(cleaned, profile) or self._heuristic_intent(corrected_text, profile, raw_cleaned)
        prediction = self.model.predict(corrected_text)
        intent = heuristic_intent or prediction.intent
        confidence = 0.92 if heuristic_intent else prediction.confidence

        if is_goodbye_message(cleaned) and (
            profile.current_stage == "recommend"
            or profile.conversation_turns > 1
            or any(token in cleaned for token in {"пока", "до свидания", "до встречи", "до связи", "бывай"})
        ):
            profile.current_stage = "smalltalk"
            profile.last_bot_question = None
            return BotReply(
                text="Пока! Если захочешь вернуться к разговору или к подбору инструмента — я помогу.",
                detected_intent="goodbye",
                confidence=max(confidence, 0.97),
            )

        if profile.current_stage == "smalltalk" and sentiment <= -0.45:
            return BotReply(
                text=(
                    "Похоже, настроение не самое лёгкое. Можем без спешки просто пообщаться, "
                    "а если захочешь переключиться на музыку или подобрать что-то для души — я рядом."
                ),
                detected_intent=intent,
                confidence=max(confidence, 0.8),
            )

        if intent == "about_bot":
            return BotReply(
                text=(
                    "Я бот-помощник по музыкальным инструментам. Могу просто поддержать живой разговор, "
                    "обсудить интересы, музыку и потом — если тебе это будет реально интересно — помочь спокойно выбрать инструмент без навязывания."
                ),
                detected_intent="about_bot",
                confidence=0.97,
            )

        if profile.current_stage == "smalltalk" and is_counter_question(cleaned):
            if profile.last_bot_question == "smalltalk_probe" and is_weather_related(cleaned):
                reply = random.choice(WEATHER_COUNTER_REPLIES)
            else:
                reply = random.choice(COUNTER_QUESTION_REPLIES)
            return BotReply(
                text=reply,
                detected_intent=intent or "smalltalk",
                confidence=max(confidence, 0.84),
            )

        if profile.current_stage == "music_hook" and not self._looks_like_music_answer(cleaned):
            if is_smalltalk_request(cleaned):
                profile.current_stage = "smalltalk"
                profile.last_bot_question = None
            elif intent == "buy_interest":
                pass
            elif detect_genre(cleaned) or detect_category(cleaned) or detect_purpose(cleaned) or detect_experience_level(cleaned) or is_preference_statement(cleaned):
                pass
            else:
                profile.current_stage = "smalltalk"
                profile.last_bot_question = None

        if intent is not None:
            sales_reply = handle_sales_flow(profile, intent, corrected_text, self.catalog)
            if sales_reply:
                return BotReply(text=sales_reply, detected_intent=intent, confidence=confidence)

        if profile.current_stage == "smalltalk":
            if cleaned in {"привет", "здравствуй", "добрый вечер", "добрый день", "хай", "приветик", "привет!"}:
                profile.last_bot_question = "smalltalk_probe"
                profile.proactive_smalltalk_step = max(profile.proactive_smalltalk_step, 1)
                return BotReply(
                    text=random.choice(GREETING_WITH_QUESTION),
                    detected_intent=intent or "greeting",
                    confidence=max(confidence, 0.9),
                )

        if intent is not None:
            direct_response = self._intent_response(intent)
            if direct_response:
                if profile.current_stage == "smalltalk":
                    bridge = maybe_switch_to_music(profile, intent)
                    if bridge:
                        return BotReply(text=f"{direct_response}\n\n{bridge}", detected_intent=intent, confidence=confidence)
                return BotReply(text=direct_response, detected_intent=intent, confidence=confidence)

        if profile.current_stage == "smalltalk":
            if is_positive_smalltalk(cleaned):
                bridge = None
                if profile.conversation_turns >= 5 and not profile.music_hook_offered:
                    bridge = maybe_switch_to_music(profile, "smalltalk")
                reply = random.choice(POSITIVE_SMALLTALK_REPLIES)
                if bridge and bridge not in reply:
                    return BotReply(text=f"{reply}\n\n{bridge}", detected_intent=intent, confidence=max(confidence, 0.82))
                return BotReply(text=reply, detected_intent=intent or "smalltalk", confidence=max(confidence, 0.82))
            generated = self.retriever.search(corrected_text)
            if generated:
                bridge = maybe_switch_to_music(profile, intent or "smalltalk")
                if bridge:
                    return BotReply(text=f"{generated}\n\n{bridge}", detected_intent=intent, confidence=confidence)
                return BotReply(text=generated, detected_intent=intent, confidence=confidence)
            if profile.proactive_smalltalk_step < 2 and profile.conversation_turns >= 1 and not profile.music_hook_offered:
                question = PROACTIVE_SMALLTALK_QUESTIONS[profile.proactive_smalltalk_step % len(PROACTIVE_SMALLTALK_QUESTIONS)]
                profile.proactive_smalltalk_step += 1
                profile.last_bot_question = "smalltalk_probe"
                return BotReply(
                    text=question,
                    detected_intent=intent or "smalltalk",
                    confidence=max(confidence, 0.75),
                )
            if profile.conversation_turns >= 2 and not profile.music_hook_offered:
                return BotReply(
                    text=random.choice(PROACTIVE_SMALLTALK_PROMPTS),
                    detected_intent=intent or "smalltalk",
                    confidence=max(confidence, 0.75),
                )

        generated = self.retriever.search(corrected_text)
        if generated:
            return BotReply(text=generated, detected_intent=intent, confidence=confidence)

        context_reply = self._contextual_fallback(profile, corrected_text, prediction)
        if context_reply:
            return BotReply(text=context_reply, detected_intent=intent, confidence=confidence)

        return BotReply(
            text=(
                "Я могу поддержать обычный разговор, а ещё помочь подобрать музыкальный инструмент. "
                "Расскажи, что тебе сейчас интереснее — просто пообщаться или что-то выбрать?"
            ),
            detected_intent=intent,
            confidence=confidence,
        )

    def _intent_response(self, intent: str) -> str | None:
        data = self.intent_config["intents"].get(intent)
        if not data:
            return None
        responses = data.get("responses", [])
        return random.choice(responses) if responses else None

    def _resolve_contextual_short_intent(self, cleaned: str, profile: UserProfile) -> str | None:
        if profile.current_stage == "music_hook":
            if is_affirmative_reply(cleaned):
                return "music_yes"
            if is_negative_reply(cleaned):
                return "music_no"
        if profile.current_stage == "recommend":
            if is_affirmative_reply(cleaned):
                return "buy_interest"
            if is_negative_reply(cleaned) or is_more_request(cleaned):
                return "reject_product"
        if profile.current_stage == "smalltalk":
            if cleaned in {"привет", "здравствуй", "добрый вечер", "добрый день", "хай", "приветик"}:
                return "greeting"
            if is_goodbye_message(cleaned):
                return "goodbye"
            if cleaned in {"спасибо", "благодарю", "спс", "спасибо большое"}:
                return "thanks"
        return None

    def _looks_like_music_answer(self, cleaned: str) -> bool:
        music_markers = {
            "да",
            "ага",
            "угу",
            "слушаю",
            "слушаю регулярно",
            "да слушаю",
            "люблю",
            "интересна",
            "интересно",
            "музыка",
            "гитар",
            "укул",
            "пианино",
            "клав",
            "синт",
            "рок",
            "джаз",
            "класс",
            "электрон",
            "нович",
            "играю",
            "присматрива",
            "бюджет",
        }
        return any(marker in cleaned for marker in music_markers)

    def _contextual_fallback(self, profile: UserProfile, text: str, prediction: IntentPrediction) -> str | None:
        cleaned = clear_text(text)
        if not cleaned:
            return "Можешь написать чуть подробнее? Тогда я лучше пойму, что именно ты имеешь в виду."

        if prediction.intent == "about_bot":
            return (
                "Я бот-помощник по музыкальным инструментам. Могу просто поддержать живой разговор, "
                "обсудить интересы, музыку и потом — если тебе это будет реально интересно — помочь спокойно выбрать инструмент без навязывания."
            )

        if profile.current_stage == "smalltalk" and prediction.intent == "smalltalk":
            return "С удовольствием. Можем поговорить про музыку, фильмы, хобби, настроение или просто о том, как проходит день. Что тебе ближе сейчас?"

        if prediction.intent == "buy_interest":
            if profile.current_stage == "smalltalk":
                profile.current_stage = "music_hook"
                profile.last_bot_question = "music_interest"
                profile.discussed_music = True
                return "Давай. Чтобы не советовать наугад, сначала быстро пойму твой вкус: музыка тебе вообще близка — слушаешь что-то регулярно?"
            return handle_sales_flow(profile, "buy_interest", text, self.catalog)

        if profile.current_stage == "budget" and (any(ch.isdigit() for ch in cleaned) or "к" in cleaned or "тысяч" in cleaned or "тыс" in cleaned):
            return handle_sales_flow(profile, prediction.intent or "", text, self.catalog)

        if profile.current_stage == "recommend" and (is_short_rejection(cleaned) or "дорог" in cleaned or "не нрав" in cleaned):
            return handle_sales_flow(profile, "reject_product", text, self.catalog)

        if profile.current_stage == "recommend" and is_more_request(cleaned):
            return handle_sales_flow(profile, "reject_product", text, self.catalog)

        if profile.current_stage == "recommend" and (is_order_confirmation(cleaned) or is_current_option_choice(cleaned) or is_preference_statement(cleaned)):
            return handle_sales_flow(profile, "buy_interest", text, self.catalog)

        if profile.current_stage in {"music_hook", "experience", "soft_interest"} and detect_purpose(cleaned):
            return handle_sales_flow(profile, prediction.intent or "", text, self.catalog)

        if profile.current_stage in {"music_hook", "experience", "soft_interest"} and (
            detect_category(cleaned) or detect_genre(cleaned) or detect_experience_level(cleaned) or is_preference_statement(cleaned)
        ):
            return handle_sales_flow(profile, prediction.intent or "", text, self.catalog)

        if profile.current_stage == "recommend" and (is_short_confirmation(cleaned) or "да" == cleaned):
            return handle_sales_flow(profile, "buy_interest", text, self.catalog)

        if profile.current_stage == "budget":
            return "Я пока не увидел сумму. Напиши примерный бюджет цифрами, например 15000 или 30000."

        if profile.current_stage in {"music_hook", "experience", "soft_interest"}:
            return handle_sales_flow(profile, prediction.intent or "", text, self.catalog)

        if profile.current_stage == "smalltalk":
            if cleaned in {"ага", "угу", "мм", "хм", "ясно", "понятно", "точно", "ну да", "ладно", "окей", "ок"}:
                return random.choice(INTERJECTION_REPLIES)
            return random.choice(SMALLTALK_FALLBACKS)

        return None

    def _heuristic_intent(self, text: str, profile: UserProfile, raw_cleaned: str | None = None) -> str | None:
        cleaned = clear_text(text)
        raw = raw_cleaned or cleaned
        if not cleaned and not raw:
            return None
        detected_genre = detect_genre(cleaned) or detect_genre(raw)
        detected_category = detect_category(cleaned) or detect_category(raw)
        detected_experience = detect_experience_level(cleaned) or detect_experience_level(raw)
        haystacks = {value for value in {cleaned, raw} if value}

        if any(detect_purpose(value) for value in haystacks):
            return "instrument_interest"
        if detected_genre:
            return "genre"
        if detected_experience == "продвинутый":
            return "advanced"
        if detected_experience == "новичок":
            return "beginner"
        if any("нович" in value or "с нуля" in value or "только начинаю" in value for value in haystacks):
            return "beginner"
        if any("присматрива" in value or "пока выбираю" in value or "пока смотрю" in value for value in haystacks):
            return "beginner"
        if any("играю давно" in value or "есть опыт" in value or "не новичок" in value for value in haystacks):
            return "advanced"
        if detected_category:
            return "instrument_interest"
        if any(is_preference_statement(value) for value in haystacks) and (detected_category or detected_genre or any(detect_purpose(value) for value in haystacks)):
            return "instrument_interest"
        if profile.current_stage == "recommend" and any(any(word in value for word in {"еще", "ещё", "другой", "не нравится", "не то", "дорого"}) for value in haystacks):
            return "reject_product"
        if any("хобби" in value or "увлеч" in value for value in haystacks):
            return "hobby"
        return None
