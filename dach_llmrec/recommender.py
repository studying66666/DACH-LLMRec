from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .constants import (
    CONTENT_STATUS_SCORE,
    FEEDBACK_WEIGHTS,
    NUTRITION_TIER_SCORE,
    RECIPE_WEIGHTS,
    RECIPE_WEIGHTS_WITH_DISEASE,
)
from .embeddings import EmbeddingProvider, HashEmbeddingProvider
from .models import Ingredient, Recipe, UserProfile
from .paths import DEFAULT_DB_PATH


class DACHLLMRecommender:
    """Health-factor extensible DACH-LLMRec prototype.

    The implementation reads the current SQLite database and computes an
    explainable Top-K ranking. Disease tables are not used by default because
    the current database does not contain a reliable user-to-disease profile.
    """

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        embedding_provider: EmbeddingProvider | None = None,
        feedback_before: str | None = None,
        bpr_model_path: str | Path | None = None,
        disabled_components: set[str] | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.embedding_provider = embedding_provider or HashEmbeddingProvider()
        self.feedback_before = feedback_before
        self.disabled_components = disabled_components or set()
        self.bpr_scorer = None
        if bpr_model_path:
            from .bpr import BPRScorer

            self.bpr_scorer = BPRScorer.load(bpr_model_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        self.recipes: dict[int, Recipe] = {}
        self.ingredients: dict[int, Ingredient] = {}
        self.recipe_ingredients: dict[int, dict[int, float]] = defaultdict(dict)
        self.recipe_ingredient_sets: dict[int, set[int]] = defaultdict(set)
        self.recipe_method_sets: dict[int, set[str]] = {}
        self.recipe_main_ingredients: dict[int, set[int]] = defaultdict(set)
        self.ingredient_tastes: dict[int, set[int]] = defaultdict(set)
        self.recipe_taste_vectors: dict[int, dict[int, float]] = {}
        self.recipe_health_direct: dict[int, dict[int, float]] = defaultdict(dict)
        self.ingredient_health: dict[int, dict[int, float]] = defaultdict(dict)
        self.hci_parent: dict[int, int] = {}
        self.hci_children: dict[int, set[int]] = defaultdict(set)
        self.recipe_feedback: dict[tuple[int, int], float] = defaultdict(float)
        self.ingredient_feedback_scores: dict[tuple[int, int], float] = {}
        self.recipe_nutrition_tier: dict[int, str] = {}
        self.disease_avoid_recipes: dict[int, set[int]] = defaultdict(set)
        self.disease_avoid_ingredients: dict[int, set[int]] = defaultdict(set)
        self.disease_recommend_recipes: dict[int, dict[int, float]] = defaultdict(dict)
        self.disease_recommend_ingredients: dict[int, dict[int, float]] = defaultdict(dict)
        self._recipe_embedding_cache: dict[int, list[float]] = {}
        self._ingredient_embedding_cache: dict[int, list[float]] = {}

        self._load()

    def close(self) -> None:
        self.conn.close()

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        mode: str = "recipe",
        health_factors: list[dict[str, Any]] | None = None,
        enable_disease_constraints: bool = False,
    ) -> dict[str, Any]:
        """Return Top-K recipe or ingredient recommendations.

        health_factors may include future disease/risk inputs such as
        {"type": "disease", "id": 123, "weight": 1.0}. Disease constraints are
        disabled by default and only used when explicitly enabled.
        """

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if mode not in {"recipe", "ingredient"}:
            raise ValueError("mode must be 'recipe' or 'ingredient'")

        profile = self._load_user_profile(user_id)
        user_embedding = self.embedding_provider.embed(self._user_text(profile))
        disease_ids = self._extract_disease_ids(health_factors, enable_disease_constraints)
        learned_scores = None
        if mode == "recipe" and self.bpr_scorer is not None:
            learned_scores = self.bpr_scorer.score_many(profile.user_id, list(self.recipes))
        if mode == "recipe":
            items = self._recommend_recipes(
                profile, top_k, disease_ids, user_embedding, learned_scores
            )
        else:
            items = self._recommend_ingredients(profile, top_k, disease_ids, user_embedding)
        return {
            "user_id": user_id,
            "mode": mode,
            "items": items,
            "metadata": {
                "database": str(self.db_path),
                "llm_provider": type(self.embedding_provider).__name__,
                "disease_constraints_enabled": bool(disease_ids),
                "data_boundary": (
                    "Synthetic user and feedback tables are used for simulation; "
                    "disease tables are not treated as user diagnoses."
                ),
            },
        }

    def validate(self, user_id: int = 1, top_k: int = 10) -> dict[str, Any]:
        """Run core acceptance checks against recipe recommendations."""

        result = self.recommend(user_id=user_id, top_k=top_k, mode="recipe")
        profile = self._load_user_profile(user_id)
        violations = []
        for item in result["items"]:
            recipe_id = item["item_id"]
            recipe = self.recipes[recipe_id]
            ingredients = set(self.recipe_ingredients.get(recipe_id, {}))
            if recipe.recommendable != 1:
                violations.append({"recipe_id": recipe_id, "type": "not_recommendable"})
            if recipe_id in profile.avoided_recipes:
                violations.append({"recipe_id": recipe_id, "type": "avoided_recipe"})
            avoided_hits = sorted(ingredients & profile.avoided_ingredients)
            if avoided_hits:
                violations.append(
                    {
                        "recipe_id": recipe_id,
                        "type": "avoided_ingredient",
                        "ingredient_ids": avoided_hits,
                    }
                )
        return {
            "user_id": user_id,
            "top_k": top_k,
            "returned": len(result["items"]),
            "violations": violations,
            "passed": len(violations) == 0,
        }

    def _load(self) -> None:
        self._load_recipes()
        self._load_ingredients()
        self._load_recipe_ingredients()
        self._load_taste_links()
        self._build_recipe_taste_vectors()
        self._load_hci_hierarchy()
        self._load_hci_links()
        self._load_feedback()
        self._build_ingredient_feedback_scores()
        self._load_nutrition_tiers()
        self._load_disease_extension_tables()

    def _load_recipes(self) -> None:
        rows = self.conn.execute(
            """
            SELECT recipe_id, name, description, cuisine_name, cooking_methods,
                   taste_tags, content_status, recommendable, restriction_reasons
            FROM norm_recipe_v1
            """
        )
        for row in rows:
            recipe_id = int(row["recipe_id"])
            self.recipes[recipe_id] = Recipe(
                recipe_id=recipe_id,
                name=row["name"] or "",
                description=row["description"] or "",
                cuisine_name=row["cuisine_name"] or "",
                cooking_methods=tuple(_json_list(row["cooking_methods"])),
                taste_tags=tuple(_json_list(row["taste_tags"])),
                content_status=row["content_status"] or "",
                recommendable=int(row["recommendable"] or 0),
                restriction_reasons=row["restriction_reasons"] or "",
            )
            self.recipe_method_sets[recipe_id] = set(self.recipes[recipe_id].cooking_methods)

    def _load_ingredients(self) -> None:
        rows = self.conn.execute(
            """
            SELECT ingredient_id, name, foodtype_id, nutrition_status, source_status
            FROM norm_ingredient_v1
            WHERE ingredient_id IS NOT NULL AND ingredient_id <> -2
            """
        )
        for row in rows:
            ingredient_id = int(row["ingredient_id"])
            self.ingredients[ingredient_id] = Ingredient(
                ingredient_id=ingredient_id,
                name=row["name"] or "",
                foodtype_id=_to_int_or_none(row["foodtype_id"]),
                nutrition_status=row["nutrition_status"] or "",
                source_status=row["source_status"] or "",
            )

    def _load_recipe_ingredients(self) -> None:
        rows = self.conn.execute(
            """
            SELECT recipe_id, resolved_ingredient_id, term_ingredient_id, is_main
            FROM norm_recipe_ingredient_v1
            WHERE recipe_id IS NOT NULL
              AND is_food_input = 1
            """
        )
        for row in rows:
            recipe_id = int(row["recipe_id"])
            ingredient_id = _valid_id(row["resolved_ingredient_id"])
            if ingredient_id is None:
                ingredient_id = _valid_id(row["term_ingredient_id"])
            if ingredient_id is None:
                continue
            weight = 1.0 if int(row["is_main"] or 0) == 1 else 0.5
            self.recipe_ingredients[recipe_id][ingredient_id] = max(
                self.recipe_ingredients[recipe_id].get(ingredient_id, 0.0), weight
            )
            self.recipe_ingredient_sets[recipe_id].add(ingredient_id)
            if weight >= 1.0:
                self.recipe_main_ingredients[recipe_id].add(ingredient_id)

    def _load_taste_links(self) -> None:
        rows = self.conn.execute(
            """
            SELECT food, taste
            FROM ingredient2taste
            WHERE food IS NOT NULL AND food <> -2
              AND taste IS NOT NULL AND taste <> -2
            """
        )
        for row in rows:
            self.ingredient_tastes[int(row["food"])].add(int(row["taste"]))

    def _build_recipe_taste_vectors(self) -> None:
        for recipe_id, ingredients in self.recipe_ingredients.items():
            counter: Counter[int] = Counter()
            for ingredient_id, ingredient_weight in ingredients.items():
                for taste_id in self.ingredient_tastes.get(ingredient_id, ()):
                    counter[taste_id] += ingredient_weight
            total = sum(counter.values())
            if total:
                self.recipe_taste_vectors[recipe_id] = {
                    taste_id: value / total for taste_id, value in counter.items()
                }

    def _load_hci_links(self) -> None:
        rows = self.conn.execute(
            """
            SELECT hci, recipe, intensity
            FROM hcirecommendrecipe
            WHERE hci IS NOT NULL AND hci <> -2
              AND recipe IS NOT NULL AND recipe <> -2
            """
        )
        for row in rows:
            hci_id = int(row["hci"])
            recipe_id = int(row["recipe"])
            self.recipe_health_direct[recipe_id][hci_id] = max(
                self.recipe_health_direct[recipe_id].get(hci_id, 0.0),
                _clip01(float(row["intensity"] or 0) / 5.0),
            )

    def _load_hci_hierarchy(self) -> None:
        rows = self.conn.execute(
            """
            SELECT id, parent
            FROM hci
            WHERE id IS NOT NULL AND id <> -2
            """
        )
        for row in rows:
            hci_id = int(row["id"])
            parent_id = _valid_id(row["parent"])
            if parent_id is not None:
                self.hci_parent[hci_id] = parent_id
                self.hci_children[parent_id].add(hci_id)

        rows = self.conn.execute(
            """
            SELECT hci, ingredient, intensity
            FROM hcirecommendingredient
            WHERE hci IS NOT NULL AND hci <> -2
              AND ingredient IS NOT NULL AND ingredient <> -2
            """
        )
        for row in rows:
            hci_id = int(row["hci"])
            ingredient_id = int(row["ingredient"])
            self.ingredient_health[ingredient_id][hci_id] = max(
                self.ingredient_health[ingredient_id].get(hci_id, 0.0),
                _clip01(float(row["intensity"] or 0) / 5.0),
            )

    def _load_feedback(self) -> None:
        query = """
            SELECT user_id, recipe_id, event_type
            FROM norm_synthetic_feedback_event_v1
            WHERE user_id IS NOT NULL AND recipe_id IS NOT NULL
        """
        params: tuple[Any, ...] = ()
        if self.feedback_before:
            query += " AND event_time < ?"
            params = (self.feedback_before,)
        rows = self.conn.execute(query, params)
        for row in rows:
            event_type = row["event_type"] or ""
            self.recipe_feedback[(int(row["user_id"]), int(row["recipe_id"]))] += (
                FEEDBACK_WEIGHTS.get(event_type, 0.0)
            )

    def _build_ingredient_feedback_scores(self) -> None:
        totals: dict[tuple[int, int], float] = defaultdict(float)
        counts: dict[tuple[int, int], int] = defaultdict(int)
        for (user_id, recipe_id), _raw in self.recipe_feedback.items():
            score = self._feedback_score(user_id, recipe_id)
            for ingredient_id in self.recipe_ingredient_sets.get(recipe_id, ()):
                key = (user_id, ingredient_id)
                totals[key] += score
                counts[key] += 1
        self.ingredient_feedback_scores = {
            key: totals[key] / counts[key]
            for key in totals
            if counts[key]
        }

    def _load_nutrition_tiers(self) -> None:
        rows = self.conn.execute(
            """
            SELECT recipe_id, nutrition_feature_tier
            FROM norm_recipe_nutrition_feature_eligibility_v1
            WHERE recipe_id IS NOT NULL
            """
        )
        for row in rows:
            recipe_id = _to_int_or_none(row["recipe_id"])
            if recipe_id is not None:
                self.recipe_nutrition_tier[recipe_id] = row["nutrition_feature_tier"] or ""

    def _load_disease_extension_tables(self) -> None:
        for disease_id, recipe_id in self.conn.execute(
            """
            SELECT disease, recipe
            FROM diseaseavoidrecipe
            WHERE disease IS NOT NULL AND disease <> -2
              AND recipe IS NOT NULL AND recipe <> -2
            """
        ):
            self.disease_avoid_recipes[int(disease_id)].add(int(recipe_id))

        for disease_id, ingredient_id in self.conn.execute(
            """
            SELECT disease, ingredient
            FROM diseaseavoidingredient
            WHERE disease IS NOT NULL AND disease <> -2
              AND ingredient IS NOT NULL AND ingredient <> -2
            """
        ):
            self.disease_avoid_ingredients[int(disease_id)].add(int(ingredient_id))

        for disease_id, recipe_id, intensity in self.conn.execute(
            """
            SELECT disease, recipe, intensity
            FROM diseaserecommendrecipe
            WHERE disease IS NOT NULL AND disease <> -2
              AND recipe IS NOT NULL AND recipe <> -2
            """
        ):
            self.disease_recommend_recipes[int(disease_id)][int(recipe_id)] = _clip01(
                float(intensity or 0) / 5.0
            )

        for disease_id, ingredient_id, intensity in self.conn.execute(
            """
            SELECT disease, ingredient, intensity
            FROM diseaserecommendingredient
            WHERE disease IS NOT NULL AND disease <> -2
              AND ingredient IS NOT NULL AND ingredient <> -2
            """
        ):
            self.disease_recommend_ingredients[int(disease_id)][int(ingredient_id)] = _clip01(
                float(intensity or 0) / 5.0
            )

    def _load_user_profile(self, user_id: int) -> UserProfile:
        row = self.conn.execute(
            """
            SELECT user_id, age_years, sex, activity_level, diet_goal
            FROM norm_synthetic_user_v1
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"user_id {user_id} not found in norm_synthetic_user_v1")

        profile = UserProfile(
            user_id=int(row["user_id"]),
            age_years=_to_int_or_none(row["age_years"]),
            sex=row["sex"] or "",
            activity_level=row["activity_level"] or "",
            diet_goal=row["diet_goal"] or "",
        )

        for taste_id, preference in self.conn.execute(
            """
            SELECT taste_id, preference
            FROM norm_synthetic_user_taste_v1
            WHERE user_id = ? AND taste_id IS NOT NULL AND taste_id <> -2
            """,
            (user_id,),
        ):
            profile.taste_preferences[int(taste_id)] = float(preference or 0) / 2.0

        for hci_id, priority in self.conn.execute(
            """
            SELECT hci_id, priority
            FROM norm_synthetic_user_health_goal_v1
            WHERE user_id = ? AND hci_id IS NOT NULL AND hci_id <> -2
            """,
            (user_id,),
        ):
            priority_value = max(float(priority or 1), 1.0)
            profile.health_goals[int(hci_id)] = max(
                profile.health_goals.get(int(hci_id), 0.0), 1.0 / priority_value
            )
        profile.health_goals = self._expand_health_goals(profile.health_goals)

        sport_rows = self.conn.execute(
            """
            SELECT sport_id, sessions_per_week, minutes_per_session, intensity
            FROM norm_synthetic_user_sport_v1
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchall()
        if sport_rows:
            minutes = sum(
                float(row["sessions_per_week"] or 0) * float(row["minutes_per_session"] or 0)
                for row in sport_rows
            )
            intensities = ",".join(sorted({row["intensity"] or "" for row in sport_rows}))
            profile.sport_summary = f"weekly_minutes={minutes:.0f}; intensity={intensities}"

        aliases = _user_id_aliases(user_id)
        placeholders = ",".join("?" for _ in aliases)
        for ingredient_id, intensity in self.conn.execute(
            f"""
            SELECT ingredient, intensity
            FROM userfondnessingredient
            WHERE user IN ({placeholders})
              AND ingredient IS NOT NULL AND ingredient <> -2
            """,
            aliases,
        ):
            score = _clip01(float(intensity or 0) / 5.0)
            profile.favored_ingredients[int(ingredient_id)] = max(
                profile.favored_ingredients.get(int(ingredient_id), 0.0), score
            )

        for ingredient_id, in self.conn.execute(
            f"""
            SELECT ingredient
            FROM useravoidingredient
            WHERE user IN ({placeholders})
              AND ingredient IS NOT NULL AND ingredient <> -2
            """,
            aliases,
        ):
            profile.avoided_ingredients.add(int(ingredient_id))

        for recipe_id, in self.conn.execute(
            f"""
            SELECT recipe
            FROM useravoidrecipe
            WHERE user IN ({placeholders})
              AND recipe IS NOT NULL AND recipe <> -2
            """,
            aliases,
        ):
            profile.avoided_recipes.add(int(recipe_id))

        return profile

    def _expand_health_goals(self, goals: dict[int, float]) -> dict[int, float]:
        expanded = dict(goals)
        for hci_id, weight in list(goals.items()):
            for descendant_id in self._hci_descendants(hci_id):
                expanded[descendant_id] = max(expanded.get(descendant_id, 0.0), weight * 0.8)
            parent_id = self.hci_parent.get(hci_id)
            while parent_id is not None:
                expanded[parent_id] = max(expanded.get(parent_id, 0.0), weight * 0.5)
                parent_id = self.hci_parent.get(parent_id)
        return expanded

    def _hci_descendants(self, hci_id: int) -> set[int]:
        descendants: set[int] = set()
        stack = list(self.hci_children.get(hci_id, set()))
        while stack:
            child_id = stack.pop()
            if child_id in descendants:
                continue
            descendants.add(child_id)
            stack.extend(self.hci_children.get(child_id, set()))
        return descendants

    def _recommend_recipes(
        self,
        profile: UserProfile,
        top_k: int,
        disease_ids: list[int],
        user_embedding: list[float] | None = None,
        learned_scores: dict[int, float] | None = None,
    ) -> list[dict[str, Any]]:
        scored: list[dict[str, Any]] = []
        for recipe_id, recipe in self.recipes.items():
            if not self._passes_recipe_filters(recipe_id, profile, disease_ids):
                continue
            evidence = self._recipe_evidence(
                profile, recipe_id, disease_ids, user_embedding, learned_scores
            )
            weights = RECIPE_WEIGHTS_WITH_DISEASE if disease_ids else RECIPE_WEIGHTS
            score = self._weighted_score(evidence, weights)
            scored.append({"recipe_id": recipe_id, "score": score, "evidence": evidence})

        selected = self._apply_diversity(profile, scored, top_k, disease_ids)
        items = []
        for row in selected:
            recipe = self.recipes[row["recipe_id"]]
            evidence = row["evidence"]
            matched = self._matched_factors(evidence, include_disease=bool(disease_ids))
            items.append(
                {
                    "item_id": recipe.recipe_id,
                    "item_type": "recipe",
                    "name": recipe.name,
                    "score": round(row["score"], 6),
                    "evidence": {key: round(value, 6) for key, value in evidence.items()},
                    "matched_factors": matched,
                    "explanation": self._recipe_explanation(recipe, evidence, matched),
                }
            )
        return items

    def _recommend_ingredients(
        self,
        profile: UserProfile,
        top_k: int,
        disease_ids: list[int],
        user_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        items = []
        if user_embedding is None:
            user_embedding = self.embedding_provider.embed(self._user_text(profile))
        for ingredient_id, ingredient in self.ingredients.items():
            if ingredient_id in profile.avoided_ingredients:
                continue
            if any(ingredient_id in self.disease_avoid_ingredients[disease_id] for disease_id in disease_ids):
                continue
            evidence = self._ingredient_evidence(
                profile, ingredient_id, user_embedding, disease_ids
            )
            score = (
                0.25 * evidence["preference_score"]
                + 0.25 * evidence["health_goal_score"]
                + 0.20 * evidence["content_score"]
                + 0.10 * evidence["feedback_score"]
                + 0.10 * evidence["llm_alignment_score"]
                + 0.10 * evidence["quality_score"]
            )
            if disease_ids:
                score = 0.18 * evidence["preference_score"] + 0.18 * evidence[
                    "health_goal_score"
                ] + 0.18 * evidence["disease_score"] + 0.18 * evidence[
                    "content_score"
                ] + 0.08 * evidence[
                    "feedback_score"
                ] + 0.10 * evidence[
                    "llm_alignment_score"
                ] + 0.10 * evidence[
                    "quality_score"
                ]
            matched = self._matched_factors(evidence, include_disease=bool(disease_ids))
            items.append(
                {
                    "item_id": ingredient.ingredient_id,
                    "item_type": "ingredient",
                    "name": ingredient.name,
                    "score": round(score, 6),
                    "evidence": {key: round(value, 6) for key, value in evidence.items()},
                    "matched_factors": matched,
                    "explanation": self._ingredient_explanation(ingredient, matched),
                }
            )
        items.sort(key=lambda item: item["score"], reverse=True)
        return items[:top_k]

    def _passes_recipe_filters(
        self,
        recipe_id: int,
        profile: UserProfile,
        disease_ids: list[int],
    ) -> bool:
        recipe = self.recipes[recipe_id]
        if recipe.recommendable != 1:
            return False
        if recipe_id in profile.avoided_recipes:
            return False
        ingredients = self.recipe_ingredient_sets.get(recipe_id, set())
        if ingredients & profile.avoided_ingredients:
            return False
        for disease_id in disease_ids:
            if recipe_id in self.disease_avoid_recipes[disease_id]:
                return False
            if ingredients & self.disease_avoid_ingredients[disease_id]:
                return False
        return True

    def _recipe_evidence(
        self,
        profile: UserProfile,
        recipe_id: int,
        disease_ids: list[int],
        user_embedding: list[float] | None = None,
        learned_scores: dict[int, float] | None = None,
    ) -> dict[str, float]:
        recipe = self.recipes[recipe_id]
        if user_embedding is None:
            user_embedding = self.embedding_provider.embed(self._user_text(profile))
        return {
            "preference_score": self._taste_score(profile, recipe_id),
            "health_goal_score": self._health_goal_score(profile, recipe_id),
            "disease_score": self._disease_score(recipe_id, disease_ids),
            "content_score": self._recipe_content_score(profile, recipe_id),
            "feedback_score": self._feedback_score(
                profile.user_id, recipe_id, learned_scores.get(recipe_id) if learned_scores else None
            ),
            "llm_alignment_score": _cosine01(
                user_embedding, self._recipe_embedding(recipe)
            ),
            "quality_score": self._recipe_quality_score(recipe_id),
            "diversity_boost": 0.5,
        } | self._disabled_evidence_overrides()

    def _disabled_evidence_overrides(self) -> dict[str, float]:
        overrides = {}
        if "preference" in self.disabled_components:
            overrides["preference_score"] = 0.5
        if "health" in self.disabled_components:
            overrides["health_goal_score"] = 0.0
        if "content" in self.disabled_components:
            overrides["content_score"] = 0.5
        if "feedback" in self.disabled_components:
            overrides["feedback_score"] = 0.5
        if "llm" in self.disabled_components:
            overrides["llm_alignment_score"] = 0.5
        if "quality" in self.disabled_components:
            overrides["quality_score"] = 0.5
        if "diversity" in self.disabled_components:
            overrides["diversity_boost"] = 0.5
        return overrides

    def _ingredient_evidence(
        self,
        profile: UserProfile,
        ingredient_id: int,
        user_embedding: list[float],
        disease_ids: list[int],
    ) -> dict[str, float]:
        ingredient = self.ingredients[ingredient_id]
        taste_score = self._ingredient_taste_score(profile, ingredient_id)
        health_score = self._ingredient_health_goal_score(profile, ingredient_id)
        disease_score = self._ingredient_disease_score(ingredient_id, disease_ids)
        content_score = profile.favored_ingredients.get(ingredient_id, 0.5)
        feedback_score = self._ingredient_feedback_score(profile.user_id, ingredient_id)
        llm_score = _cosine01(user_embedding, self._ingredient_embedding(ingredient))
        quality_score = 1.0 if ingredient.nutrition_status == "observed" else 0.6
        return {
            "preference_score": taste_score,
            "health_goal_score": health_score,
            "disease_score": disease_score,
            "content_score": content_score,
            "feedback_score": feedback_score,
            "llm_alignment_score": llm_score,
            "quality_score": quality_score,
        }

    def _taste_score(self, profile: UserProfile, recipe_id: int) -> float:
        if not profile.taste_preferences:
            return 0.5
        recipe_vector = self.recipe_taste_vectors.get(recipe_id)
        if not recipe_vector:
            return 0.5
        raw = _sparse_cosine(profile.taste_preferences, recipe_vector)
        return (raw + 1.0) / 2.0

    def _ingredient_taste_score(self, profile: UserProfile, ingredient_id: int) -> float:
        tastes = self.ingredient_tastes.get(ingredient_id)
        if not profile.taste_preferences or not tastes:
            return 0.5
        values = [profile.taste_preferences.get(taste_id, 0.0) for taste_id in tastes]
        return _clip01((sum(values) / len(values) + 1.0) / 2.0)

    def _health_goal_score(self, profile: UserProfile, recipe_id: int) -> float:
        if not profile.health_goals:
            return 0.5
        direct = 0.0
        for hci_id, priority_weight in profile.health_goals.items():
            direct = max(
                direct,
                priority_weight * self.recipe_health_direct.get(recipe_id, {}).get(hci_id, 0.0),
            )

        ingredient_scores = []
        for ingredient_id in self.recipe_ingredient_sets.get(recipe_id, set()):
            best = 0.0
            for hci_id, priority_weight in profile.health_goals.items():
                best = max(
                    best,
                    priority_weight * self.ingredient_health.get(ingredient_id, {}).get(hci_id, 0.0),
                )
            ingredient_scores.append(best)
        indirect = sum(ingredient_scores) / len(ingredient_scores) if ingredient_scores else 0.0
        return _clip01(0.6 * direct + 0.4 * indirect)

    def _ingredient_health_goal_score(self, profile: UserProfile, ingredient_id: int) -> float:
        if not profile.health_goals:
            return 0.5
        best = 0.0
        for hci_id, priority_weight in profile.health_goals.items():
            best = max(
                best,
                priority_weight * self.ingredient_health.get(ingredient_id, {}).get(hci_id, 0.0),
            )
        return _clip01(best)

    def _disease_score(self, recipe_id: int, disease_ids: list[int]) -> float:
        if not disease_ids:
            return 0.0
        direct = max(
            (
                self.disease_recommend_recipes[disease_id].get(recipe_id, 0.0)
                for disease_id in disease_ids
            ),
            default=0.0,
        )
        ingredient_scores = []
        for ingredient_id in self.recipe_ingredients.get(recipe_id, {}):
            best = max(
                (
                    self.disease_recommend_ingredients[disease_id].get(ingredient_id, 0.0)
                    for disease_id in disease_ids
                ),
                default=0.0,
            )
            ingredient_scores.append(best)
        indirect = sum(ingredient_scores) / len(ingredient_scores) if ingredient_scores else 0.0
        return _clip01(0.6 * direct + 0.4 * indirect)

    def _ingredient_disease_score(self, ingredient_id: int, disease_ids: list[int]) -> float:
        if not disease_ids:
            return 0.0
        return max(
            (
                self.disease_recommend_ingredients[disease_id].get(ingredient_id, 0.0)
                for disease_id in disease_ids
            ),
            default=0.0,
        )

    def _recipe_content_score(self, profile: UserProfile, recipe_id: int) -> float:
        recipe_ingredients = self.recipe_ingredients.get(recipe_id, {})
        if not profile.favored_ingredients or not recipe_ingredients:
            return 0.5
        numerator = 0.0
        denominator = sum(recipe_ingredients.values()) or 1.0
        for ingredient_id, recipe_weight in recipe_ingredients.items():
            numerator += recipe_weight * profile.favored_ingredients.get(ingredient_id, 0.0)
        return _clip01(numerator / denominator)

    def _feedback_score(
        self,
        user_id: int,
        recipe_id: int,
        learned_score: float | None = None,
    ) -> float:
        raw = self.recipe_feedback.get((user_id, recipe_id))
        event_score = 0.5 if raw is None else 1.0 / (1.0 + math.exp(-raw / 5.0))
        if self.bpr_scorer is None and learned_score is None:
            return event_score
        if learned_score is None and self.bpr_scorer is not None:
            learned_score = self.bpr_scorer.score(user_id, recipe_id)
        if learned_score is None:
            return event_score
        return _clip01(0.5 * event_score + 0.5 * learned_score)

    def _ingredient_feedback_score(self, user_id: int, ingredient_id: int) -> float:
        return self.ingredient_feedback_scores.get((user_id, ingredient_id), 0.5)

    def _recipe_quality_score(self, recipe_id: int) -> float:
        recipe = self.recipes[recipe_id]
        content_score = CONTENT_STATUS_SCORE.get(recipe.content_status, 0.5)
        nutrition_score = NUTRITION_TIER_SCORE.get(
            self.recipe_nutrition_tier.get(recipe_id, ""), 0.5
        )
        return _clip01(content_score * nutrition_score)

    def _apply_diversity(
        self,
        profile: UserProfile,
        scored: list[dict[str, Any]],
        top_k: int,
        disease_ids: list[int],
    ) -> list[dict[str, Any]]:
        weights = RECIPE_WEIGHTS_WITH_DISEASE if disease_ids else RECIPE_WEIGHTS
        pool = sorted(scored, key=lambda row: row["score"], reverse=True)[: max(200, top_k * 25)]
        selected: list[dict[str, Any]] = []
        while pool and len(selected) < top_k:
            best_index = 0
            best_score = -1.0
            for index, row in enumerate(pool):
                diversity = (
                    0.5
                    if "diversity" in self.disabled_components
                    else self._diversity_boost(row["recipe_id"], [x["recipe_id"] for x in selected])
                )
                evidence = dict(row["evidence"])
                evidence["diversity_boost"] = diversity
                score = self._weighted_score(evidence, weights)
                if score > best_score:
                    best_index = index
                    best_score = score
            chosen = pool.pop(best_index)
            chosen["evidence"] = dict(chosen["evidence"])
            chosen["evidence"]["diversity_boost"] = (
                0.5
                if "diversity" in self.disabled_components
                else self._diversity_boost(chosen["recipe_id"], [x["recipe_id"] for x in selected])
            )
            chosen["score"] = self._weighted_score(chosen["evidence"], weights)
            selected.append(chosen)
        return selected

    def _diversity_boost(self, recipe_id: int, selected_recipe_ids: list[int]) -> float:
        if not selected_recipe_ids:
            return 1.0
        recipe = self.recipes[recipe_id]
        recipe_methods = self.recipe_method_sets.get(recipe_id, set(recipe.cooking_methods))
        recipe_main = self.recipe_main_ingredients.get(recipe_id, set())
        max_similarity = 0.0
        for selected_id in selected_recipe_ids:
            selected = self.recipes[selected_id]
            similarity = 0.0
            if recipe.cuisine_name and recipe.cuisine_name == selected.cuisine_name:
                similarity += 0.4
            similarity += 0.3 * _jaccard(
                recipe_methods,
                self.recipe_method_sets.get(selected_id, set(selected.cooking_methods)),
            )
            similarity += 0.3 * _jaccard(
                recipe_main, self.recipe_main_ingredients.get(selected_id, set())
            )
            max_similarity = max(max_similarity, similarity)
        return _clip01(1.0 - max_similarity)

    def _weighted_score(self, evidence: dict[str, float], weights: dict[str, float]) -> float:
        return _clip01(
            weights["preference"] * evidence["preference_score"]
            + weights["health_goal"] * evidence["health_goal_score"]
            + weights.get("disease", 0.0) * evidence.get("disease_score", 0.0)
            + weights["content"] * evidence["content_score"]
            + weights["feedback"] * evidence["feedback_score"]
            + weights["llm_alignment"] * evidence["llm_alignment_score"]
            + weights["quality"] * evidence["quality_score"]
            + weights["diversity"] * evidence["diversity_boost"]
        )

    def _recipe_embedding(self, recipe: Recipe) -> list[float]:
        if recipe.recipe_id not in self._recipe_embedding_cache:
            ingredient_names = [
                self.ingredients[ingredient_id].name
                for ingredient_id in self.recipe_ingredients.get(recipe.recipe_id, {})
                if ingredient_id in self.ingredients
            ][:30]
            text = "；".join(
                [
                    f"菜谱:{recipe.name}",
                    f"描述:{recipe.description}",
                    f"菜系:{recipe.cuisine_name}",
                    f"做法:{','.join(recipe.cooking_methods)}",
                    f"口味:{','.join(recipe.taste_tags)}",
                    f"食材:{','.join(ingredient_names)}",
                    f"营养可信度:{self.recipe_nutrition_tier.get(recipe.recipe_id, '')}",
                ]
            )
            self._recipe_embedding_cache[recipe.recipe_id] = self.embedding_provider.embed(text)
        return self._recipe_embedding_cache[recipe.recipe_id]

    def _ingredient_embedding(self, ingredient: Ingredient) -> list[float]:
        if ingredient.ingredient_id not in self._ingredient_embedding_cache:
            text = "；".join(
                [
                    f"食材:{ingredient.name}",
                    f"类别:{ingredient.foodtype_id or ''}",
                    f"营养状态:{ingredient.nutrition_status}",
                    f"来源:{ingredient.source_status}",
                ]
            )
            self._ingredient_embedding_cache[ingredient.ingredient_id] = (
                self.embedding_provider.embed(text)
            )
        return self._ingredient_embedding_cache[ingredient.ingredient_id]

    def _user_text(self, profile: UserProfile) -> str:
        health_goals = ",".join(str(hci_id) for hci_id in sorted(profile.health_goals))
        tastes = ",".join(
            f"{taste_id}:{score:.1f}"
            for taste_id, score in sorted(profile.taste_preferences.items())
        )
        favored = ",".join(
            self.ingredients[ingredient_id].name
            for ingredient_id in sorted(profile.favored_ingredients)[:20]
            if ingredient_id in self.ingredients
        )
        return "；".join(
            [
                f"年龄:{profile.age_years or ''}",
                f"性别:{profile.sex}",
                f"活动水平:{profile.activity_level}",
                f"饮食目标:{profile.diet_goal}",
                f"口味:{tastes}",
                f"健康目标:{health_goals}",
                f"运动:{profile.sport_summary}",
                f"偏好食材:{favored}",
            ]
        )

    def _matched_factors(
        self,
        evidence: dict[str, float],
        include_disease: bool = False,
    ) -> list[str]:
        labels = []
        thresholds = {
            "preference_score": (0.6, "口味"),
            "health_goal_score": (0.2, "健康目标"),
            "content_score": (0.2, "偏好食材"),
            "feedback_score": (0.6, "历史反馈"),
            "llm_alignment_score": (0.6, "语义匹配"),
            "quality_score": (0.6, "内容质量"),
        }
        for key, (threshold, label) in thresholds.items():
            if evidence.get(key, 0.0) >= threshold:
                labels.append(label)
        if include_disease and evidence.get("disease_score", 0.0) >= 0.2:
            labels.append("疾病/风险约束")
        return labels

    def _recipe_explanation(
        self,
        recipe: Recipe,
        evidence: dict[str, float],
        matched: list[str],
    ) -> str:
        if not matched:
            return f"推荐《{recipe.name}》是因为它通过了安全过滤，并在综合排序中得分较高。"
        parts = "、".join(matched)
        return f"推荐《{recipe.name}》是因为它通过了安全过滤，并匹配了{parts}等已计算证据。"

    def _ingredient_explanation(self, ingredient: Ingredient, matched: list[str]) -> str:
        if not matched:
            return f"推荐“{ingredient.name}”是因为它通过了避免食材过滤，并在综合排序中得分较高。"
        parts = "、".join(matched)
        return f"推荐“{ingredient.name}”是因为它通过了避免食材过滤，并匹配了{parts}等已计算证据。"

    def _extract_disease_ids(
        self,
        health_factors: list[dict[str, Any]] | None,
        enable_disease_constraints: bool,
    ) -> list[int]:
        if not enable_disease_constraints or not health_factors:
            return []
        disease_ids = []
        for factor in health_factors:
            if factor.get("type") == "disease":
                disease_id = _to_int_or_none(factor.get("id"))
                if disease_id is not None and disease_id != -2:
                    disease_ids.append(disease_id)
        return disease_ids


def _valid_id(value: Any) -> int | None:
    parsed = _to_int_or_none(value)
    if parsed is None or parsed == -2:
        return None
    return parsed


def _to_int_or_none(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return [value]
    if isinstance(parsed, list):
        return [str(item) for item in parsed if item not in (None, "", -2)]
    return [str(parsed)]


def _user_id_aliases(user_id: int) -> tuple[int, ...]:
    aliases = {user_id}
    if 1 <= user_id <= 500:
        aliases.add(1_000_000 + user_id)
    return tuple(sorted(aliases))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _sparse_cosine(left: dict[int, float], right: dict[int, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0.0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _cosine01(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.5
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.5
    return _clip01((dot / (left_norm * right_norm) + 1.0) / 2.0)


def _jaccard(left: set[Any], right: set[Any]) -> float:
    if not left and not right:
        return 0.0
    return len(left & right) / len(left | right)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run DACH-LLMRec recommendations.")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite database path")
    parser.add_argument("--user-id", type=int, default=1, help="Synthetic user id")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    parser.add_argument("--mode", choices=["recipe", "ingredient"], default="recipe")
    parser.add_argument("--validate", action="store_true", help="Run validation instead")
    parser.add_argument("--bpr-model", default=None, help="Optional trained BPR .pt artifact")
    parser.add_argument(
        "--disease-id",
        action="append",
        type=int,
        default=[],
        help="Optional disease id for future disease constraints",
    )
    args = parser.parse_args(argv)

    recommender = DACHLLMRecommender(args.db, bpr_model_path=args.bpr_model)
    try:
        if args.validate:
            output = recommender.validate(user_id=args.user_id, top_k=args.top_k)
        else:
            health_factors = [
                {"type": "disease", "id": disease_id} for disease_id in args.disease_id
            ]
            output = recommender.recommend(
                user_id=args.user_id,
                top_k=args.top_k,
                mode=args.mode,
                health_factors=health_factors,
                enable_disease_constraints=bool(args.disease_id),
            )
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    finally:
        recommender.close()


if __name__ == "__main__":
    raise SystemExit(main())
