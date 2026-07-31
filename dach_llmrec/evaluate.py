from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from .constants import FEEDBACK_WEIGHTS
from .itemknn import ItemKNNScorer
from .paths import DEFAULT_DB_PATH
from .recommender import DACHLLMRecommender


POSITIVE_EVENTS = {"click", "save", "cook"}


def evaluate(
    db_path: str | Path = DEFAULT_DB_PATH,
    cutoff: str = "2026-06-01 00:00:00",
    top_k: int = 10,
    max_users: int | None = 50,
    bpr_model_path: str | Path | None = None,
    rankers: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate on synthetic feedback after cutoff.

    This is a simulation sanity check, not evidence from real users. The model
    loads only feedback before cutoff, then positive events after cutoff are
    treated as held-out positives.
    """

    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        test_positives = _load_test_positives(conn, cutoff)
        user_ids = sorted(test_positives)
        if max_users is not None:
            user_ids = user_ids[:max_users]
        popularity = _training_popularity(conn, cutoff)
        rankers = rankers or [
            "popularity",
            "content",
            "itemknn",
            "bpr_only",
            "dach_no_health",
            "dach_no_llm",
            "dach_no_feedback",
            "dach_no_diversity",
            "dach_full",
        ]
        results = {}
        for ranker in rankers:
            if ranker == "bpr_only" and not bpr_model_path:
                results[ranker] = {"skipped": True, "reason": "bpr_model_path is required"}
                continue
            recommender = DACHLLMRecommender(
                db_path,
                feedback_before=cutoff,
                bpr_model_path=bpr_model_path if ranker in {"bpr_only", "dach_full", "dach_no_health", "dach_no_llm", "dach_no_feedback", "dach_no_diversity"} else None,
                disabled_components=_disabled_components_for_ranker(ranker),
            )
            try:
                results[ranker] = _evaluate_ranker(
                    recommender=recommender,
                    user_ids=user_ids,
                    test_positives=test_positives,
                    top_k=top_k,
                    ranker=ranker,
                    popularity=popularity,
                )
            finally:
                recommender.close()
        return {
            "metadata": {
                "database": str(db_path),
                "cutoff": cutoff,
                "top_k": top_k,
                "evaluated_users": len(user_ids),
                "bpr_model": str(bpr_model_path) if bpr_model_path else None,
                "boundary": "synthetic feedback simulation; not real-user validation",
            },
            "results": results,
        }
    finally:
        conn.close()


def _load_test_positives(conn: sqlite3.Connection, cutoff: str) -> dict[int, set[int]]:
    rows = conn.execute(
        """
        SELECT user_id, recipe_id
        FROM norm_synthetic_feedback_event_v1
        WHERE event_time >= ?
          AND event_type IN ('click', 'save', 'cook')
          AND user_id IS NOT NULL AND recipe_id IS NOT NULL
        """,
        (cutoff,),
    )
    positives: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        positives[int(row["user_id"])].add(int(row["recipe_id"]))
    return positives


def _training_popularity(conn: sqlite3.Connection, cutoff: str) -> list[int]:
    scores: dict[int, float] = defaultdict(float)
    rows = conn.execute(
        """
        SELECT recipe_id, event_type
        FROM norm_synthetic_feedback_event_v1
        WHERE event_time < ?
          AND user_id IS NOT NULL AND recipe_id IS NOT NULL
        """,
        (cutoff,),
    )
    for row in rows:
        scores[int(row["recipe_id"])] += FEEDBACK_WEIGHTS.get(row["event_type"], 0.0)
    return [
        recipe_id
        for recipe_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ]


def _evaluate_ranker(
    recommender: DACHLLMRecommender,
    user_ids: list[int],
    test_positives: dict[int, set[int]],
    top_k: int,
    ranker: str,
    popularity: list[int] | None = None,
) -> dict[str, float]:
    precision_values = []
    recall_values = []
    ndcg_values = []
    hit_values = []
    safety_violations = 0
    returned = 0
    catalog_hits: set[int] = set()
    diversity_values = []
    itemknn_scorer = None
    itemknn_candidate_recipe_ids: list[int] | None = None
    if ranker == "itemknn":
        itemknn_scorer = ItemKNNScorer.from_feedback(
            recipe_ids=sorted(recommender.recipes),
            recipe_feedback=recommender.recipe_feedback,
        )
        itemknn_candidate_recipe_ids = sorted(itemknn_scorer.recipe_to_index)

    for user_id in user_ids:
        positives = test_positives[user_id]
        if ranker == "dach":
            items = recommender.recommend(user_id=user_id, top_k=top_k, mode="recipe")["items"]
            rec_ids = [item["item_id"] for item in items]
        elif ranker == "popularity":
            rec_ids = _popularity_for_user(recommender, user_id, popularity or [], top_k)
        elif ranker == "content":
            rec_ids = _content_for_user(recommender, user_id, top_k)
        elif ranker == "itemknn":
            rec_ids = _itemknn_for_user(
                recommender,
                itemknn_scorer,
                user_id,
                top_k,
                itemknn_candidate_recipe_ids or [],
            )
        elif ranker == "bpr_only":
            rec_ids = _bpr_only_for_user(recommender, user_id, top_k)
        else:
            items = recommender.recommend(user_id=user_id, top_k=top_k, mode="recipe")["items"]
            rec_ids = [item["item_id"] for item in items]

        hits = [recipe_id for recipe_id in rec_ids if recipe_id in positives]
        precision_values.append(len(hits) / max(len(rec_ids), 1))
        recall_values.append(len(hits) / max(len(positives), 1))
        hit_values.append(1.0 if hits else 0.0)
        ndcg_values.append(_ndcg(rec_ids, positives, top_k))
        catalog_hits.update(rec_ids)
        returned += len(rec_ids)
        diversity_values.append(_intra_list_diversity(recommender, rec_ids))
        safety_violations += _safety_violations_for_rec_ids(recommender, user_id, rec_ids)

    total_recipes = max(len(recommender.recipes), 1)
    return {
        "precision_at_k": _mean(precision_values),
        "recall_at_k": _mean(recall_values),
        "ndcg_at_k": _mean(ndcg_values),
        "hit_rate_at_k": _mean(hit_values),
        "coverage": len(catalog_hits) / total_recipes,
        "diversity": _mean(diversity_values),
        "safety_violation_rate": safety_violations / max(returned, 1),
    }


def _disabled_components_for_ranker(ranker: str) -> set[str]:
    mapping = {
        "dach_no_health": {"health"},
        "dach_no_llm": {"llm"},
        "dach_no_feedback": {"feedback"},
        "dach_no_diversity": {"diversity"},
    }
    return mapping.get(ranker, set())


def _popularity_for_user(
    recommender: DACHLLMRecommender,
    user_id: int,
    popularity: list[int],
    top_k: int,
) -> list[int]:
    profile = recommender._load_user_profile(user_id)
    rec_ids = []
    for recipe_id in popularity:
        if recommender._passes_recipe_filters(recipe_id, profile, []):
            rec_ids.append(recipe_id)
        if len(rec_ids) >= top_k:
            break
    return rec_ids


def _content_for_user(
    recommender: DACHLLMRecommender,
    user_id: int,
    top_k: int,
) -> list[int]:
    profile = recommender._load_user_profile(user_id)
    scored = []
    for recipe_id, recipe in recommender.recipes.items():
        if not recommender._passes_recipe_filters(recipe_id, profile, []):
            continue
        evidence = recommender._recipe_evidence(profile, recipe_id, [])
        score = (
            0.45 * evidence["preference_score"]
            + 0.35 * evidence["content_score"]
            + 0.20 * evidence["quality_score"]
        )
        scored.append((recipe_id, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [recipe_id for recipe_id, _ in scored[:top_k]]


def _bpr_only_for_user(
    recommender: DACHLLMRecommender,
    user_id: int,
    top_k: int,
) -> list[int]:
    if recommender.bpr_scorer is None:
        return []
    seen_recipe_ids = {
        recipe_id
        for seen_user_id, recipe_id in recommender.recipe_feedback
        if seen_user_id == user_id
    }
    candidate_recipe_ids = [
        recipe_id
        for recipe_id in recommender.recipes
        if recipe_id in recommender.bpr_scorer.recipe_to_index
    ]
    return recommender.bpr_scorer.topk(
        user_id=user_id,
        top_k=top_k,
        candidate_recipe_ids=candidate_recipe_ids,
        exclude_recipe_ids=seen_recipe_ids,
    )


def _itemknn_for_user(
    recommender: DACHLLMRecommender,
    itemknn_scorer: ItemKNNScorer | None,
    user_id: int,
    top_k: int,
    candidate_recipe_ids: list[int],
) -> list[int]:
    if itemknn_scorer is None:
        return []
    profile = recommender._load_user_profile(user_id)
    seen_recipe_ids = {
        recipe_id
        for seen_user_id, recipe_id in recommender.recipe_feedback
        if seen_user_id == user_id
    }
    filtered_candidate_recipe_ids = [
        recipe_id
        for recipe_id in candidate_recipe_ids
        if recommender._passes_recipe_filters(recipe_id, profile, [])
    ]
    return itemknn_scorer.topk(
        user_id=user_id,
        top_k=top_k,
        candidate_recipe_ids=filtered_candidate_recipe_ids,
        exclude_recipe_ids=seen_recipe_ids,
    )


def _safety_violations_for_rec_ids(
    recommender: DACHLLMRecommender,
    user_id: int,
    rec_ids: list[int],
) -> int:
    profile = recommender._load_user_profile(user_id)
    violations = 0
    for recipe_id in rec_ids:
        recipe = recommender.recipes[recipe_id]
        ingredients = set(recommender.recipe_ingredients.get(recipe_id, {}))
        if recipe.recommendable != 1:
            violations += 1
        if recipe_id in profile.avoided_recipes:
            violations += 1
        if ingredients & profile.avoided_ingredients:
            violations += 1
    return violations


def _ndcg(rec_ids: list[int], positives: set[int], top_k: int) -> float:
    dcg = 0.0
    for index, recipe_id in enumerate(rec_ids[:top_k]):
        if recipe_id in positives:
            dcg += 1.0 / math.log2(index + 2)
    ideal_hits = min(len(positives), top_k)
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def _intra_list_diversity(recommender: DACHLLMRecommender, rec_ids: list[int]) -> float:
    if len(rec_ids) < 2:
        return 0.0
    distances = []
    for left_index, left_id in enumerate(rec_ids):
        left_recipe = recommender.recipes[left_id]
        left_ingredients = set(recommender.recipe_ingredients.get(left_id, {}))
        for right_id in rec_ids[left_index + 1 :]:
            right_recipe = recommender.recipes[right_id]
            right_ingredients = set(recommender.recipe_ingredients.get(right_id, {}))
            same_cuisine = 1.0 if left_recipe.cuisine_name == right_recipe.cuisine_name else 0.0
            ingredient_overlap = (
                len(left_ingredients & right_ingredients) / len(left_ingredients | right_ingredients)
                if left_ingredients or right_ingredients
                else 0.0
            )
            similarity = 0.5 * same_cuisine + 0.5 * ingredient_overlap
            distances.append(1.0 - similarity)
    return _mean(distances)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate DACH-LLMRec on synthetic feedback.")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite database path")
    parser.add_argument("--cutoff", default="2026-06-01 00:00:00", help="Temporal split cutoff")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=50)
    parser.add_argument("--bpr-model", default=None, help="Optional trained BPR .pt artifact")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument(
        "--rankers",
        default=None,
        help="Comma-separated rankers. Defaults to all baselines and ablations.",
    )
    args = parser.parse_args(argv)
    output = evaluate(
        db_path=args.db,
        cutoff=args.cutoff,
        top_k=args.top_k,
        max_users=args.max_users,
        bpr_model_path=args.bpr_model,
        rankers=args.rankers.split(",") if args.rankers else None,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
