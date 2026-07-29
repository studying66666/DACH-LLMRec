from __future__ import annotations


FEEDBACK_WEIGHTS = {
    "cook": 5.0,
    "save": 4.0,
    "click": 2.0,
    "impression": 0.5,
    "skip": -1.0,
    "dislike": -4.0,
}


RECIPE_WEIGHTS = {
    "preference": 0.22,
    "health_goal": 0.22,
    "content": 0.16,
    "feedback": 0.15,
    "llm_alignment": 0.10,
    "quality": 0.10,
    "diversity": 0.05,
}


RECIPE_WEIGHTS_WITH_DISEASE = {
    "preference": 0.18,
    "health_goal": 0.18,
    "disease": 0.16,
    "content": 0.14,
    "feedback": 0.12,
    "llm_alignment": 0.10,
    "quality": 0.08,
    "diversity": 0.04,
}


CONTENT_STATUS_SCORE = {
    "complete": 1.0,
    "partial": 0.6,
    "sparse": 0.3,
}


NUTRITION_TIER_SCORE = {
    "standard": 1.0,
    "sensitivity_only": 0.6,
    "exclude_from_nutrition_model": 0.2,
}
