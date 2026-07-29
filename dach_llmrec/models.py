from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Recipe:
    recipe_id: int
    name: str
    description: str
    cuisine_name: str
    cooking_methods: tuple[str, ...]
    taste_tags: tuple[str, ...]
    content_status: str
    recommendable: int
    restriction_reasons: str


@dataclass(frozen=True)
class Ingredient:
    ingredient_id: int
    name: str
    foodtype_id: int | None
    nutrition_status: str
    source_status: str


@dataclass
class UserProfile:
    user_id: int
    age_years: int | None = None
    sex: str = ""
    activity_level: str = ""
    diet_goal: str = ""
    taste_preferences: dict[int, float] = field(default_factory=dict)
    health_goals: dict[int, float] = field(default_factory=dict)
    sport_summary: str = ""
    favored_ingredients: dict[int, float] = field(default_factory=dict)
    avoided_ingredients: set[int] = field(default_factory=set)
    avoided_recipes: set[int] = field(default_factory=set)
