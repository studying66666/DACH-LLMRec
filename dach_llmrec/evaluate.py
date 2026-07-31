from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from .constants import FEEDBACK_WEIGHTS
from .embeddings import (
    DEFAULT_EMBEDDING_CACHE_DIR,
    DEFAULT_REAL_EMBEDDING_MODEL,
    build_embedding_provider,
)
from .als import ALSScorer
from .fusion import FusionScorer, fit_recipe_fusion_scorer
from .itemknn import ItemKNNScorer
from .paths import DEFAULT_DB_PATH
from .recommender import DACHLLMRecommender
from .weight_search import GridSearchWeightScorer, fit_grid_search_weight_scorer


POSITIVE_EVENTS = {"click", "save", "cook"}


def evaluate(
    db_path: str | Path = DEFAULT_DB_PATH,
    cutoff: str = "2026-06-01 00:00:00",
    top_k: int = 10,
    max_users: int | None = 50,
    bpr_model_path: str | Path | None = None,
    augmented_bpr_model_path: str | Path | None = None,
    rankers: list[str] | None = None,
    embedding_provider: str = "hash",
    embedding_model: str = DEFAULT_REAL_EMBEDDING_MODEL,
    embedding_device: str = "auto",
    embedding_cache_dir: str | Path | None = DEFAULT_EMBEDDING_CACHE_DIR,
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
            "content_feedback",
            "itemknn",
            "als_only",
            "bpr_only",
            "llmrec_aug_bpr",
            "fusion_lr",
            "dach_no_semantic",
            "dach_hash_embedding",
            "dach_real_embedding",
            "dach_grid",
            "dach_no_health",
            "dach_no_llm",
            "dach_no_feedback",
            "dach_no_diversity",
            "dach_full",
        ]
        results = {}
        model_summaries: dict[str, Any] = {}
        for ranker in rankers:
            if ranker == "bpr_only" and not bpr_model_path:
                results[ranker] = {"skipped": True, "reason": "bpr_model_path is required"}
                continue
            if ranker == "llmrec_aug_bpr" and not augmented_bpr_model_path:
                results[ranker] = {"skipped": True, "reason": "augmented_bpr_model_path is required"}
                continue
            if ranker == "dach_real_embedding" and embedding_provider != "real":
                results[ranker] = {
                    "skipped": True,
                    "reason": "embedding_provider=real is required to run real embedding ablation",
                }
                continue
            selected_bpr_model_path = _bpr_model_for_ranker(
                ranker,
                bpr_model_path=bpr_model_path,
                augmented_bpr_model_path=augmented_bpr_model_path,
            )
            selected_embedding_provider = _embedding_provider_name_for_ranker(
                ranker, embedding_provider
            )
            try:
                selected_provider = build_embedding_provider(
                    embedding_provider=selected_embedding_provider,
                    embedding_model=embedding_model,
                    embedding_device=embedding_device,
                    embedding_cache_dir=embedding_cache_dir,
                )
                recommender = DACHLLMRecommender(
                    db_path,
                    embedding_provider=selected_provider,
                    feedback_before=cutoff,
                    bpr_model_path=selected_bpr_model_path,
                    disabled_components=_disabled_components_for_ranker(ranker),
                )
            except Exception as exc:
                results[ranker] = {"skipped": True, "reason": str(exc)}
                continue
            try:
                fusion_scorer = None
                grid_scorer = None
                if ranker == "fusion_lr":
                    try:
                        fusion_scorer, fusion_summary = fit_recipe_fusion_scorer(
                            recommender=recommender,
                            cutoff=cutoff,
                            max_users=max_users,
                        )
                        model_summaries[ranker] = fusion_summary
                    except ValueError as exc:
                        results[ranker] = {"skipped": True, "reason": str(exc)}
                        continue
                if ranker == "dach_grid":
                    try:
                        grid_scorer, grid_summary = fit_grid_search_weight_scorer(
                            recommender=recommender,
                            test_positives=test_positives,
                            user_ids=user_ids,
                            top_k=top_k,
                        )
                        model_summaries[ranker] = grid_summary
                    except ValueError as exc:
                        results[ranker] = {"skipped": True, "reason": str(exc)}
                        continue
                results[ranker] = _evaluate_ranker(
                    recommender=recommender,
                    user_ids=user_ids,
                    test_positives=test_positives,
                    top_k=top_k,
                    ranker=ranker,
                    popularity=popularity,
                    fusion_scorer=fusion_scorer,
                    grid_scorer=grid_scorer,
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
                "augmented_bpr_model": str(augmented_bpr_model_path) if augmented_bpr_model_path else None,
                "embedding_config": {
                    "provider": embedding_provider,
                    "model": embedding_model,
                    "device": embedding_device,
                    "cache_dir": str(embedding_cache_dir) if embedding_cache_dir else None,
                },
                "boundary": "synthetic feedback simulation; not real-user validation",
            },
            "results": results,
            "models": model_summaries,
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
    fusion_scorer: FusionScorer | None = None,
    grid_scorer: GridSearchWeightScorer | None = None,
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
    als_scorer = None
    als_candidate_recipe_ids: list[int] | None = None
    if ranker == "itemknn":
        itemknn_scorer = ItemKNNScorer.from_feedback(
            recipe_ids=sorted(recommender.recipes),
            recipe_feedback=recommender.recipe_feedback,
        )
        itemknn_candidate_recipe_ids = sorted(itemknn_scorer.recipe_to_index)
    if ranker in {"als", "als_only"}:
        als_scorer = ALSScorer.from_feedback(
            recipe_ids=sorted(recommender.recipes),
            recipe_feedback=recommender.recipe_feedback,
        )
        als_candidate_recipe_ids = sorted(als_scorer.recipe_to_index)

    for user_id in user_ids:
        positives = test_positives[user_id]
        if ranker == "dach":
            items = recommender.recommend(user_id=user_id, top_k=top_k, mode="recipe")["items"]
            rec_ids = [item["item_id"] for item in items]
        elif ranker == "popularity":
            rec_ids = _popularity_for_user(recommender, user_id, popularity or [], top_k)
        elif ranker == "content":
            rec_ids = _content_for_user(recommender, user_id, top_k)
        elif ranker == "content_feedback":
            rec_ids = _content_feedback_for_user(recommender, user_id, top_k)
        elif ranker == "itemknn":
            rec_ids = _itemknn_for_user(
                recommender,
                itemknn_scorer,
                user_id,
                top_k,
                itemknn_candidate_recipe_ids or [],
            )
        elif ranker in {"als", "als_only"}:
            rec_ids = _als_for_user(
                recommender,
                als_scorer,
                user_id,
                top_k,
                als_candidate_recipe_ids or [],
            )
        elif ranker in {"bpr_only", "llmrec_aug_bpr"}:
            rec_ids = _bpr_only_for_user(recommender, user_id, top_k)
        elif ranker == "fusion_lr":
            rec_ids = _fusion_lr_for_user(
                recommender,
                fusion_scorer,
                user_id,
                top_k,
            )
        elif ranker == "dach_grid":
            rec_ids = _dach_grid_for_user(
                recommender,
                grid_scorer,
                user_id,
                top_k,
            )
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


def _bpr_model_for_ranker(
    ranker: str,
    bpr_model_path: str | Path | None,
    augmented_bpr_model_path: str | Path | None,
) -> str | Path | None:
    if ranker == "llmrec_aug_bpr":
        return augmented_bpr_model_path
    if ranker in {
        "bpr_only",
        "fusion_lr",
        "dach_grid",
        "dach_full",
        "dach_no_health",
        "dach_no_llm",
        "dach_no_feedback",
        "dach_no_diversity",
    }:
        return bpr_model_path
    return None


def _embedding_provider_name_for_ranker(ranker: str, embedding_provider: str) -> str:
    if ranker == "dach_hash_embedding":
        return "hash"
    if ranker == "dach_real_embedding":
        return "real"
    return embedding_provider


def _disabled_components_for_ranker(ranker: str) -> set[str]:
    mapping = {
        "dach_no_health": {"health"},
        "dach_no_llm": {"llm"},
        "dach_no_semantic": {"llm"},
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


def _content_feedback_for_user(
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
        content_score = (
            0.45 * evidence["preference_score"]
            + 0.35 * evidence["content_score"]
            + 0.20 * evidence["quality_score"]
        )
        score = 0.80 * content_score + 0.20 * evidence["feedback_score"]
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
    profile = recommender._load_user_profile(user_id)
    seen_recipe_ids = {
        recipe_id
        for seen_user_id, recipe_id in recommender.recipe_feedback
        if seen_user_id == user_id
    }
    candidate_recipe_ids = [
        recipe_id
        for recipe_id in recommender.recipes
        if recipe_id in recommender.bpr_scorer.recipe_to_index
        and recommender._passes_recipe_filters(recipe_id, profile, [])
    ]
    return recommender.bpr_scorer.topk(
        user_id=user_id,
        top_k=top_k,
        candidate_recipe_ids=candidate_recipe_ids,
        exclude_recipe_ids=seen_recipe_ids,
    )

def _fusion_lr_for_user(
    recommender: DACHLLMRecommender,
    fusion_scorer: FusionScorer | None,
    user_id: int,
    top_k: int,
) -> list[int]:
    if fusion_scorer is None:
        return []
    profile = recommender._load_user_profile(user_id)
    user_embedding = recommender.embedding_provider.embed(recommender._user_text(profile))
    learned_scores = None
    if recommender.bpr_scorer is not None:
        learned_scores = recommender.bpr_scorer.score_many(user_id, list(recommender.recipes))
    seen_recipe_ids = {
        recipe_id
        for seen_user_id, recipe_id in recommender.recipe_feedback
        if seen_user_id == user_id
    }
    scored = []
    for recipe_id, recipe in recommender.recipes.items():
        if recipe_id in seen_recipe_ids:
            continue
        if not recommender._passes_recipe_filters(recipe_id, profile, []):
            continue
        evidence = recommender._recipe_evidence(
            profile, recipe_id, [], user_embedding, learned_scores
        )
        score = fusion_scorer.score(evidence)
        scored.append((recipe_id, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [recipe_id for recipe_id, _ in scored[:top_k]]


def _dach_grid_for_user(
    recommender: DACHLLMRecommender,
    grid_scorer: GridSearchWeightScorer | None,
    user_id: int,
    top_k: int,
) -> list[int]:
    if grid_scorer is None:
        return []
    return grid_scorer.topk(recommender, user_id=user_id, top_k=top_k)


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


def _als_for_user(
    recommender: DACHLLMRecommender,
    als_scorer: ALSScorer | None,
    user_id: int,
    top_k: int,
    candidate_recipe_ids: list[int],
) -> list[int]:
    if als_scorer is None:
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
    return als_scorer.topk(
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
    parser.add_argument("--augmented-bpr-model", default=None, help="Optional augmented BPR .pt artifact")
    parser.add_argument("--embedding-provider", choices=["hash", "real"], default="hash")
    parser.add_argument("--embedding-model", default=DEFAULT_REAL_EMBEDDING_MODEL)
    parser.add_argument("--embedding-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--embedding-cache-dir", default=str(DEFAULT_EMBEDDING_CACHE_DIR))
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
        augmented_bpr_model_path=args.augmented_bpr_model,
        rankers=args.rankers.split(",") if args.rankers else None,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        embedding_cache_dir=args.embedding_cache_dir,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
