from __future__ import annotations

import random
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .constants import FEEDBACK_WEIGHTS
from .recommender import DACHLLMRecommender


POSITIVE_EVENTS = {"click", "save", "cook"}
NEGATIVE_EVENTS = {"skip", "dislike"}
FEATURE_NAMES = (
    "preference_score",
    "health_goal_score",
    "disease_score",
    "content_score",
    "feedback_score",
    "llm_alignment_score",
    "quality_score",
)


@dataclass
class FusionScorer:
    feature_names: tuple[str, ...]
    model: Pipeline

    def score(self, evidence: dict[str, float]) -> float:
        features = np.asarray(
            [[float(evidence.get(name, 0.0)) for name in self.feature_names]],
            dtype=np.float64,
        )
        return float(self.model.predict_proba(features)[0, 1])

    def topk(
        self,
        recommender: DACHLLMRecommender,
        user_id: int,
        top_k: int = 10,
        candidate_recipe_ids: list[int] | None = None,
        exclude_recipe_ids: set[int] | None = None,
    ) -> list[int]:
        profile = recommender._load_user_profile(user_id)
        user_embedding = recommender.embedding_provider.embed(recommender._user_text(profile))
        learned_scores = None
        if recommender.bpr_scorer is not None:
            learned_scores = recommender.bpr_scorer.score_many(user_id, list(recommender.recipes))
        candidate_recipe_ids = candidate_recipe_ids or sorted(recommender.recipes)
        exclude_recipe_ids = exclude_recipe_ids or set()
        scored: list[tuple[int, float]] = []
        for recipe_id in candidate_recipe_ids:
            if recipe_id in exclude_recipe_ids:
                continue
            if not recommender._passes_recipe_filters(recipe_id, profile, []):
                continue
            evidence = recommender._recipe_evidence(
                profile, recipe_id, [], user_embedding, learned_scores
            )
            scored.append((recipe_id, self.score(evidence)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [recipe_id for recipe_id, _ in scored[:top_k]]

    def coefficients(self) -> dict[str, float]:
        classifier = self.model.named_steps["classifier"]
        if not hasattr(classifier, "coef_"):
            return {name: 0.0 for name in self.feature_names}
        return {
            name: float(value)
            for name, value in zip(self.feature_names, classifier.coef_[0].tolist())
        }


def fit_recipe_fusion_scorer(
    recommender: DACHLLMRecommender,
    cutoff: str,
    max_users: int | None = None,
    negative_samples_per_positive: int = 2,
    seed: int = 42,
) -> tuple[FusionScorer, dict[str, Any]]:
    rng = random.Random(seed)
    conn = recommender.conn
    positives: dict[int, set[int]] = defaultdict(set)
    negatives: dict[int, set[int]] = defaultdict(set)
    seen_items: dict[int, set[int]] = defaultdict(set)
    rows = conn.execute(
        """
        SELECT user_id, recipe_id, event_type
        FROM norm_synthetic_feedback_event_v1
        WHERE event_time < ?
          AND user_id IS NOT NULL
          AND recipe_id IS NOT NULL
          AND recipe_id <> -2
        """,
        (cutoff,),
    )
    for row in rows:
        user_id = int(row["user_id"])
        recipe_id = int(row["recipe_id"])
        event_type = row["event_type"] or ""
        seen_items[user_id].add(recipe_id)
        if event_type in POSITIVE_EVENTS or FEEDBACK_WEIGHTS.get(event_type, 0.0) > 1.0:
            positives[user_id].add(recipe_id)
        elif event_type in NEGATIVE_EVENTS or FEEDBACK_WEIGHTS.get(event_type, 0.0) < 0.0:
            negatives[user_id].add(recipe_id)

    user_ids = sorted(set(positives) | set(negatives))
    if max_users is not None:
        user_ids = user_ids[:max_users]

    candidate_recipe_ids = [
        recipe_id
        for recipe_id, recipe in sorted(recommender.recipes.items())
        if recipe.recommendable == 1
    ]

    feature_rows: list[list[float]] = []
    labels: list[int] = []
    training_user_ids: list[int] = []
    skipped_users = 0
    for user_id in user_ids:
        try:
            profile = recommender._load_user_profile(user_id)
        except ValueError:
            skipped_users += 1
            continue

        user_embedding = recommender.embedding_provider.embed(recommender._user_text(profile))
        learned_scores = None
        if recommender.bpr_scorer is not None:
            learned_scores = recommender.bpr_scorer.score_many(user_id, list(recommender.recipes))

        user_positive_items = positives.get(user_id, set())
        user_negative_items = negatives.get(user_id, set())
        user_seen_items = seen_items.get(user_id, set())

        for recipe_id in user_positive_items:
            if recipe_id not in recommender.recipes:
                continue
            feature_rows.append(
                _feature_vector(recommender, profile, recipe_id, user_embedding, learned_scores)
            )
            labels.append(1)
        for recipe_id in user_negative_items:
            if recipe_id not in recommender.recipes:
                continue
            feature_rows.append(
                _feature_vector(recommender, profile, recipe_id, user_embedding, learned_scores)
            )
            labels.append(0)

        if user_positive_items:
            pool = [recipe_id for recipe_id in candidate_recipe_ids if recipe_id not in user_seen_items]
            if pool:
                sample_size = min(
                    len(user_positive_items) * negative_samples_per_positive,
                    len(pool),
                )
                for recipe_id in rng.sample(pool, sample_size):
                    feature_rows.append(
                        _feature_vector(recommender, profile, recipe_id, user_embedding, learned_scores)
                    )
                    labels.append(0)

        if user_positive_items or user_negative_items:
            training_user_ids.append(user_id)

    if len(set(labels)) < 2:
        raise ValueError("Fusion scorer requires both positive and negative training examples.")

    features = np.asarray(feature_rows, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int32)
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(features, target)
    scorer = FusionScorer(feature_names=FEATURE_NAMES, model=model)

    classifier = model.named_steps["classifier"]
    summary = {
        "feature_names": list(FEATURE_NAMES),
        "training_users": len(training_user_ids),
        "skipped_users": skipped_users,
        "samples": int(target.shape[0]),
        "positives": int(target.sum()),
        "negatives": int(target.shape[0] - target.sum()),
        "coefficients": scorer.coefficients(),
        "intercept": float(classifier.intercept_[0]),
        "boundary": "trained on synthetic pre-cutoff feedback; not real-user validation",
    }
    return scorer, summary


def _feature_vector(
    recommender: DACHLLMRecommender,
    profile: Any,
    recipe_id: int,
    user_embedding: list[float] | None = None,
    learned_scores: dict[int, float] | None = None,
) -> list[float]:
    evidence = recommender._recipe_evidence(
        profile, recipe_id, [], user_embedding, learned_scores
    )
    return [float(evidence.get(name, 0.0)) for name in FEATURE_NAMES]

