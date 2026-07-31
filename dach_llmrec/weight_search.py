from __future__ import annotations

import argparse
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .constants import RECIPE_WEIGHTS
from .paths import DEFAULT_DB_PATH
from .recommender import DACHLLMRecommender


POSITIVE_EVENTS = {"click", "save", "cook"}
WEIGHT_FEATURES = (
    ("preference", "preference_score"),
    ("health_goal", "health_goal_score"),
    ("content", "content_score"),
    ("feedback", "feedback_score"),
    ("llm_alignment", "llm_alignment_score"),
    ("quality", "quality_score"),
    ("diversity", "diversity_boost"),
)


@dataclass
class CandidateBatch:
    recipe_ids: list[int]
    features: np.ndarray


@dataclass
class GridSearchWeightScorer:
    weights: dict[str, float]
    feature_names: tuple[str, ...]
    candidate_cache: dict[int, CandidateBatch]
    candidate_pool_size: int

    def topk(
        self,
        recommender: DACHLLMRecommender,
        user_id: int,
        top_k: int = 10,
    ) -> list[int]:
        batch = self.candidate_cache.get(user_id)
        if batch is None or not batch.recipe_ids:
            return []
        return _rank_batch(
            recommender=recommender,
            batch=batch,
            weights=self.weights,
            top_k=top_k,
            candidate_pool_size=self.candidate_pool_size,
        )


def fit_grid_search_weight_scorer(
    recommender: DACHLLMRecommender,
    test_positives: dict[int, set[int]],
    user_ids: list[int],
    top_k: int = 10,
    grid_step: float = 0.2,
    max_component_weight: float = 0.6,
    candidate_pool_size: int | None = None,
    exclude_seen: bool = True,
) -> tuple[GridSearchWeightScorer, dict[str, Any]]:
    """Select recipe fusion weights by validation-set NDCG@K grid search."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    candidate_pool_size = candidate_pool_size or max(200, top_k * 25)
    user_ids = [user_id for user_id in user_ids if user_id in test_positives]
    candidate_cache = _build_candidate_cache(
        recommender=recommender,
        user_ids=user_ids,
        exclude_seen=exclude_seen,
    )
    searchable_user_ids = [
        user_id
        for user_id in user_ids
        if user_id in candidate_cache and candidate_cache[user_id].recipe_ids
    ]
    if not searchable_user_ids:
        raise ValueError("Grid search requires at least one validation user with candidates.")

    weight_candidates = generate_weight_grid(
        step=grid_step,
        max_component_weight=max_component_weight,
        include_weights=[RECIPE_WEIGHTS],
    )
    default_weights = _normalized_weights(RECIPE_WEIGHTS)
    default_metrics = _evaluate_weights(
        recommender=recommender,
        candidate_cache=candidate_cache,
        test_positives=test_positives,
        user_ids=searchable_user_ids,
        weights=default_weights,
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
    )

    best_weights = default_weights
    best_metrics = default_metrics
    candidate_summaries: list[dict[str, Any]] = []
    for weights in weight_candidates:
        metrics = _evaluate_weights(
            recommender=recommender,
            candidate_cache=candidate_cache,
            test_positives=test_positives,
            user_ids=searchable_user_ids,
            weights=weights,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
        )
        candidate_summaries.append(
            {
                "weights": weights,
                "ndcg_at_k": metrics["ndcg_at_k"],
                "recall_at_k": metrics["recall_at_k"],
                "precision_at_k": metrics["precision_at_k"],
                "source": "default" if weights == default_weights else "grid",
            }
        )
        if _is_better(metrics, best_metrics):
            best_weights = weights
            best_metrics = metrics

    scorer = GridSearchWeightScorer(
        weights=best_weights,
        feature_names=tuple(feature_name for _weight_name, feature_name in WEIGHT_FEATURES),
        candidate_cache=candidate_cache,
        candidate_pool_size=candidate_pool_size,
    )
    candidate_summaries.sort(
        key=lambda row: (
            row["ndcg_at_k"],
            row["recall_at_k"],
            row["precision_at_k"],
            1 if row["source"] == "default" else 0,
        ),
        reverse=True,
    )
    summary = {
        "selection_metric": f"ndcg@{top_k}",
        "grid_step": grid_step,
        "max_component_weight": max_component_weight,
        "candidate_pool_size": candidate_pool_size,
        "exclude_seen": exclude_seen,
        "validation_users": len(searchable_user_ids),
        "candidate_weight_count": len(weight_candidates),
        "feature_names": list(scorer.feature_names),
        "default_weights": default_weights,
        "default_validation_metrics": default_metrics,
        "best_weights": best_weights,
        "best_validation_metrics": best_metrics,
        "best_validation_ndcg_at_k": best_metrics["ndcg_at_k"],
        "top_weight_candidates": candidate_summaries[:10],
        "boundary": "weights selected on synthetic validation feedback; not real-user validation",
    }
    return scorer, summary


def grid_search_recipe_weights(
    db_path: str | Path = DEFAULT_DB_PATH,
    cutoff: str = "2026-06-01 00:00:00",
    top_k: int = 10,
    max_users: int | None = 500,
    bpr_model_path: str | Path | None = None,
    grid_step: float = 0.2,
    max_component_weight: float = 0.6,
    candidate_pool_size: int | None = None,
    exclude_seen: bool = True,
) -> dict[str, Any]:
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        test_positives = _load_test_positives(conn, cutoff)
        user_ids = sorted(test_positives)
        if max_users is not None:
            user_ids = user_ids[:max_users]
    finally:
        conn.close()

    recommender = DACHLLMRecommender(
        db_path,
        feedback_before=cutoff,
        bpr_model_path=bpr_model_path,
    )
    try:
        _scorer, summary = fit_grid_search_weight_scorer(
            recommender=recommender,
            test_positives=test_positives,
            user_ids=user_ids,
            top_k=top_k,
            grid_step=grid_step,
            max_component_weight=max_component_weight,
            candidate_pool_size=candidate_pool_size,
            exclude_seen=exclude_seen,
        )
    finally:
        recommender.close()

    return {
        "metadata": {
            "database": str(db_path),
            "cutoff": cutoff,
            "top_k": top_k,
            "max_users": max_users,
            "bpr_model": str(bpr_model_path) if bpr_model_path else None,
        },
        "grid_search": summary,
    }


def generate_weight_grid(
    step: float = 0.2,
    max_component_weight: float = 0.6,
    include_weights: list[dict[str, float]] | None = None,
) -> list[dict[str, float]]:
    if step <= 0.0 or step > 1.0:
        raise ValueError("step must be in (0, 1]")
    total_units = round(1.0 / step)
    if not math.isclose(total_units * step, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("step must evenly divide 1.0")
    max_units = max(1, round(max_component_weight / step))
    weight_names = [name for name, _feature_name in WEIGHT_FEATURES]
    seen: set[tuple[float, ...]] = set()
    candidates: list[dict[str, float]] = []

    def add_candidate(values: list[float]) -> None:
        rounded = tuple(round(value, 10) for value in values)
        if rounded in seen:
            return
        seen.add(rounded)
        candidates.append(dict(zip(weight_names, rounded)))

    def walk(index: int, remaining: int, units: list[int]) -> None:
        if index == len(weight_names) - 1:
            if 0 <= remaining <= max_units:
                add_candidate([unit * step for unit in [*units, remaining]])
            return
        upper = min(max_units, remaining)
        for unit in range(upper + 1):
            walk(index + 1, remaining - unit, [*units, unit])

    walk(0, total_units, [])
    for weights in include_weights or []:
        add_candidate([_normalized_weights(weights).get(name, 0.0) for name in weight_names])
    return candidates


def _build_candidate_cache(
    recommender: DACHLLMRecommender,
    user_ids: list[int],
    exclude_seen: bool,
) -> dict[int, CandidateBatch]:
    cache: dict[int, CandidateBatch] = {}
    recipe_ids = list(recommender.recipes)
    feature_names = [feature_name for _weight_name, feature_name in WEIGHT_FEATURES]
    for user_id in user_ids:
        try:
            profile = recommender._load_user_profile(user_id)
        except ValueError:
            continue
        user_embedding = recommender.embedding_provider.embed(recommender._user_text(profile))
        learned_scores = None
        if recommender.bpr_scorer is not None:
            learned_scores = recommender.bpr_scorer.score_many(user_id, recipe_ids)
        seen_recipe_ids = _seen_recipe_ids(recommender, user_id) if exclude_seen else set()

        batch_recipe_ids: list[int] = []
        feature_rows: list[list[float]] = []
        for recipe_id in recipe_ids:
            if recipe_id in seen_recipe_ids:
                continue
            if not recommender._passes_recipe_filters(recipe_id, profile, []):
                continue
            evidence = recommender._recipe_evidence(
                profile, recipe_id, [], user_embedding, learned_scores
            )
            batch_recipe_ids.append(recipe_id)
            feature_rows.append([float(evidence.get(name, 0.0)) for name in feature_names])
        if not batch_recipe_ids:
            continue
        cache[user_id] = CandidateBatch(
            recipe_ids=batch_recipe_ids,
            features=np.asarray(feature_rows, dtype=np.float32),
        )
    return cache


def _evaluate_weights(
    recommender: DACHLLMRecommender,
    candidate_cache: dict[int, CandidateBatch],
    test_positives: dict[int, set[int]],
    user_ids: list[int],
    weights: dict[str, float],
    top_k: int,
    candidate_pool_size: int,
) -> dict[str, float]:
    precision_values = []
    recall_values = []
    ndcg_values = []
    hit_values = []
    returned = 0
    catalog_hits: set[int] = set()
    for user_id in user_ids:
        batch = candidate_cache.get(user_id)
        if batch is None:
            continue
        rec_ids = _rank_batch(
            recommender=recommender,
            batch=batch,
            weights=weights,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
        )
        positives = test_positives[user_id]
        hits = [recipe_id for recipe_id in rec_ids if recipe_id in positives]
        precision_values.append(len(hits) / max(len(rec_ids), 1))
        recall_values.append(len(hits) / max(len(positives), 1))
        hit_values.append(1.0 if hits else 0.0)
        ndcg_values.append(_ndcg(rec_ids, positives, top_k))
        returned += len(rec_ids)
        catalog_hits.update(rec_ids)

    total_recipes = max(len(recommender.recipes), 1)
    return {
        "precision_at_k": _mean(precision_values),
        "recall_at_k": _mean(recall_values),
        "ndcg_at_k": _mean(ndcg_values),
        "hit_rate_at_k": _mean(hit_values),
        "coverage": len(catalog_hits) / total_recipes,
        "returned": float(returned),
    }


def _rank_batch(
    recommender: DACHLLMRecommender,
    batch: CandidateBatch,
    weights: dict[str, float],
    top_k: int,
    candidate_pool_size: int,
) -> list[int]:
    if not batch.recipe_ids:
        return []
    weight_vector = np.asarray(
        [weights[name] for name, _feature_name in WEIGHT_FEATURES],
        dtype=np.float32,
    )
    diversity_weight = float(weights["diversity"])
    base_scores = batch.features @ weight_vector
    pool_size = min(len(batch.recipe_ids), max(candidate_pool_size, top_k))
    if pool_size <= 0:
        return []
    if pool_size < len(batch.recipe_ids):
        pool_indices = np.argpartition(-base_scores, pool_size - 1)[:pool_size].tolist()
        pool_indices.sort(key=lambda index: float(base_scores[index]), reverse=True)
    else:
        pool_indices = np.argsort(-base_scores).tolist()

    static_scores = base_scores - diversity_weight * batch.features[:, -1]
    selected_recipe_ids: list[int] = []
    while pool_indices and len(selected_recipe_ids) < top_k:
        best_position = 0
        best_score = float("-inf")
        for position, candidate_index in enumerate(pool_indices):
            recipe_id = batch.recipe_ids[candidate_index]
            diversity_boost = recommender._diversity_boost(recipe_id, selected_recipe_ids)
            score = float(static_scores[candidate_index]) + diversity_weight * diversity_boost
            if score > best_score:
                best_position = position
                best_score = score
        selected_index = pool_indices.pop(best_position)
        selected_recipe_ids.append(batch.recipe_ids[selected_index])
    return selected_recipe_ids


def _is_better(metrics: dict[str, float], best_metrics: dict[str, float]) -> bool:
    current_key = (
        metrics["ndcg_at_k"],
        metrics["recall_at_k"],
        metrics["precision_at_k"],
    )
    best_key = (
        best_metrics["ndcg_at_k"],
        best_metrics["recall_at_k"],
        best_metrics["precision_at_k"],
    )
    return current_key > best_key


def _normalized_weights(weights: dict[str, float]) -> dict[str, float]:
    selected = {name: max(float(weights.get(name, 0.0)), 0.0) for name, _feature in WEIGHT_FEATURES}
    total = sum(selected.values())
    if total <= 0.0:
        raise ValueError("weights must contain at least one positive component")
    return {name: value / total for name, value in selected.items()}


def _seen_recipe_ids(recommender: DACHLLMRecommender, user_id: int) -> set[int]:
    return {
        recipe_id
        for seen_user_id, recipe_id in recommender.recipe_feedback
        if seen_user_id == user_id
    }


def _load_test_positives(conn: sqlite3.Connection, cutoff: str) -> dict[int, set[int]]:
    rows = conn.execute(
        """
        SELECT user_id, recipe_id
        FROM norm_synthetic_feedback_event_v1
        WHERE event_time >= ?
          AND event_type IN ('click', 'save', 'cook')
          AND user_id IS NOT NULL AND recipe_id IS NOT NULL
          AND recipe_id <> -2
        """,
        (cutoff,),
    )
    positives: dict[int, set[int]] = {}
    for row in rows:
        positives.setdefault(int(row["user_id"]), set()).add(int(row["recipe_id"]))
    return positives


def _ndcg(rec_ids: list[int], positives: set[int], top_k: int) -> float:
    dcg = 0.0
    for index, recipe_id in enumerate(rec_ids[:top_k]):
        if recipe_id in positives:
            dcg += 1.0 / math.log2(index + 2)
    ideal_hits = min(len(positives), top_k)
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grid-search DACH-LLMRec recipe weights.")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite database path")
    parser.add_argument("--cutoff", default="2026-06-01 00:00:00")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=500)
    parser.add_argument("--bpr-model", default=None, help="Optional trained BPR .pt artifact")
    parser.add_argument("--grid-step", type=float, default=0.2)
    parser.add_argument("--max-component-weight", type=float, default=0.6)
    parser.add_argument("--candidate-pool-size", type=int, default=None)
    parser.add_argument("--include-seen", action="store_true")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args(argv)

    result = grid_search_recipe_weights(
        db_path=args.db,
        cutoff=args.cutoff,
        top_k=args.top_k,
        max_users=args.max_users,
        bpr_model_path=args.bpr_model,
        grid_step=args.grid_step,
        max_component_weight=args.max_component_weight,
        candidate_pool_size=args.candidate_pool_size,
        exclude_seen=not args.include_seen,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
