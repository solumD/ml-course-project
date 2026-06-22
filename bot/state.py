from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class UserProfile:
    name: Optional[str] = None
    music_interest: Optional[bool] = None
    experience_level: Optional[str] = None
    preferred_category: Optional[str] = None
    preferred_genre: Optional[str] = None
    budget: Optional[int] = None
    purpose: Optional[str] = None
    selected_product_id: Optional[str] = None
    last_recommended_product_ids: list[str] = field(default_factory=list)
    current_stage: str = "smalltalk"
    conversation_turns: int = 0
    last_bot_question: Optional[str] = None
    discussed_music: bool = False
    last_music_hook: Optional[str] = None
    music_hook_offered: bool = False
    direct_selection_mode: bool = False
    proactive_smalltalk_step: int = 0

    def reset(self) -> None:
        self.name = None
        self.music_interest = None
        self.experience_level = None
        self.preferred_category = None
        self.preferred_genre = None
        self.budget = None
        self.purpose = None
        self.selected_product_id = None
        self.last_recommended_product_ids.clear()
        self.current_stage = "smalltalk"
        self.conversation_turns = 0
        self.last_bot_question = None
        self.discussed_music = False
        self.last_music_hook = None
        self.music_hook_offered = False
        self.direct_selection_mode = False
        self.proactive_smalltalk_step = 0


@dataclass(slots=True)
class BotReply:
    text: str
    detected_intent: str | None = None
    confidence: float = 0.0
