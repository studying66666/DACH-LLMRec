from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from .bpr import BPRScorer
from .constants import FEEDBACK_WEIGHTS
from .paths import DEFAULT_DB_PATH
from .recommender import DACHLLMRecommender


POSITIVE_EVENTS = {"click", "save", "cook"}
NEGATIVE_EVENTS = {"skip", "dislike"}


def diagnose_bpr(
    db_path: str | Path = DEFAULT_DB_PATH,
    cutoff: str = "2026-06-01 00:00:00",
    top_k: int = 10,
    max_users: int | None = 500,
    bpr_model_path: str | Path | None = None,
) -> dict[str, Any]:
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    recommender: DACHLLMRecommender | None = None
    try:
        train = _load_training_interactions(conn, cutoff)
        test_positives = _load_test_positives(conn, cutoff)
        candidate_recipes = _load_candidate_recipes(conn)
        candidate_recipe_set = set(candidate_recipes)

        eval_user_ids = sorted(test_positives)
        if max_users is not None:
            eval_user_ids = eval_user_ids[:max_users]

        train_positive_users = set(train["positives"])
        train_negative_users = set(train["negatives"])
        train_any_users = train_positive_users | train_negative_users
        eval_users_with_training = [user_id for user_id in eval_user_ids if user_id in train_any_users]

        test_positive_items = [
            recipe_id
            for user_id in eval_user_ids
            for recipe_id in test_positives[user_id]
        ]
        test_candidate_hits = sum(1 for recipe_id in test_positive_items if recipe_id in candidate_recipe_set)

        result: dict[str, Any] = {
            "metadata": {
                "database": str(db_path),
                "cutoff": cutoff,
                "top_k": top_k,
                "max_users": max_users,
                "bpr_model": str(bpr_model_path) if bpr_model_path else None,
            },
            "training_split": {
                "users_with_positive_history": len(train_positive_users),
                "users_with_negative_history": len(train_negative_users),
                "users_with_any_history": len(train_any_users),
                "candidate_recipes": len(candidate_recipes),
                "positive_items_total": _count_items(train["positives"]),
                "negative_items_total": _count_items(train["negatives"]),
                "avg_positive_items_per_training_user": _average_count(train["positives"], train_any_users),
                "avg_negative_items_per_training_user": _average_count(train["negatives"], train_any_users),
                "negative_sampling": _negative_sampling_summary(
                    train=train,
                    candidate_recipes=candidate_recipes,
                    negative_samples_per_positive=2,
                ),
            },
            "evaluation_split": {
                "evaluated_users": len(eval_user_ids),
                "users_with_training_history": len(eval_users_with_training),
                "users_with_training_history_rate": _safe_div(
                    len(eval_users_with_training), len(eval_user_ids)
                ),
                "test_positive_items": len(test_positive_items),
                "test_positive_candidate_hits": test_candidate_hits,
                "test_positive_candidate_coverage": _safe_div(
                    test_candidate_hits, len(test_positive_items)
                ),
                **_hard_filter_coverage(db_path, eval_user_ids, test_positives),
            },
        }

        if bpr_model_path:
            recommender = DACHLLMRecommender(
                db_path,
                feedback_before=cutoff,
                bpr_model_path=bpr_model_path,
            )
            result["bpr_top_k"] = _diagnose_bpr_top_k(
                recommender=recommender,
                scorer=recommender.bpr_scorer,
                eval_user_ids=eval_user_ids,
                test_positives=test_positives,
                training_interactions=train,
                top_k=top_k,
            )
        else:
            result["bpr_top_k"] = {
                "skipped": True,
                "reason": "No --bpr-model was provided.",
            }

        return result
    finally:
        if recommender is not None:
            recommender.close()
        conn.close()


def _load_training_interactions(
    conn: sqlite3.Connection,
    cutoff: str,
) -> dict[str, dict[int, set[int]]]:
    positives: dict[int, set[int]] = defaultdict(set)
    negatives: dict[int, set[int]] = defaultdict(set)
    rows = conn.execute(
        """
        SELECT user_id, recipe_id, event_type
        FROM norm_synthetic_feedback_event_v1
        WHERE event_time < ?
          AND recipe_id IS NOT NULL AND recipe_id <> -2
        """,
        (cutoff,),
    )
    for row in rows:
        user_id = int(row["user_id"])
        recipe_id = int(row["recipe_id"])
        event_type = row["event_type"]
        if event_type in POSITIVE_EVENTS:
            positives[user_id].add(recipe_id)
        elif event_type in NEGATIVE_EVENTS:
            negatives[user_id].add(recipe_id)
        elif FEEDBACK_WEIGHTS.get(event_type, 0.0) > 1.0:
            positives[user_id].add(recipe_id)
    return {"positives": positives, "negatives": negatives}


def _load_test_positives(conn: sqlite3.Connection, cutoff: str) -> dict[int, set[int]]:
    test_positives: dict[int, set[int]] = defaultdict(set)
    rows = conn.execute(
        """
        SELECT user_id, recipe_id
        FROM norm_synthetic_feedback_event_v1
        WHERE event_time >= ?
          AND event_type IN ('click', 'save', 'cook')
          AND recipe_id IS NOT NULL AND recipe_id <> -2
        """,
        (cutoff,),
    )
    for row in rows:
        test_positives[int(row["user_id"])].add(int(row["recipe_id"]))
    return test_positives


def _load_candidate_recipes(conn: sqlite3.Connection) -> list[int]:
    rows = conn.execute(
        """
        SELECT recipe_id
        FROM norm_recipe_v1
        WHERE recommendable = 1
          AND recipe_id IS NOT NULL AND recipe_id <> -2
        ORDER BY recipe_id
        """
    )
    return [int(row["recipe_id"]) for row in rows]


def _hard_filter_coverage(
    db_path: Path,
    eval_user_ids: list[int],
    test_positives: dict[int, set[int]],
) -> dict[str, Any]:
    recommender = DACHLLMRecommender(db_path)
    try:
        total = 0
        passed = 0
        missing_profile_users: list[int] = []
        for user_id in eval_user_ids:
            try:
                profile = recommender._load_user_profile(user_id)
            except ValueError:
                missing_profile_users.append(user_id)
                continue
            for recipe_id in test_positives[user_id]:
                total += 1
                if recipe_id in recommender.recipes and recommender._passes_recipe_filters(
                    recipe_id, profile, []
                ):
                    passed += 1
        return {
            "test_positive_hard_filter_hits": passed,
            "test_positive_hard_filter_coverage": _safe_div(passed, total),
            "missing_profile_users": len(missing_profile_users),
            "missing_profile_user_examples": missing_profile_users[:10],
        }
    finally:
        recommender.close()


def _diagnose_bpr_top_k(
    recommender: DACHLLMRecommender,
    scorer: BPRScorer | None,
    eval_user_ids: list[int],
    test_positives: dict[int, set[int]],
    training_interactions: dict[str, dict[int, set[int]]],
    top_k: int,
) -> dict[str, Any]:
    if scorer is None:
        return {"skipped": True, "reason": "BPR scorer could not be loaded."}

    raw = _summarize_bpr_top_k(
        recommender=recommender,
        scorer=scorer,
        eval_user_ids=eval_user_ids,
        test_positives=test_positives,
        training_interactions=training_interactions,
        top_k=top_k,
        exclude_seen=False,
    )
    raw["exclude_seen"] = _summarize_bpr_top_k(
        recommender=recommender,
        scorer=scorer,
        eval_user_ids=eval_user_ids,
        test_positives=test_positives,
        training_interactions=training_interactions,
        top_k=top_k,
        exclude_seen=True,
    )
    return raw


def _summarize_bpr_top_k(
    recommender: DACHLLMRecommender,
    scorer: BPRScorer,
    eval_user_ids: list[int],
    test_positives: dict[int, set[int]],
    training_interactions: dict[str, dict[int, set[int]]],
    top_k: int,
    exclude_seen: bool,
) -> dict[str, Any]:
    model_user_hits = 0
    model_recipe_hits = sum(
        1 for recipe_id in recommender.recipes if recipe_id in scorer.recipe_to_index
    )
    users_with_scores = 0
    returned = 0
    hit_users = 0
    hit_items = 0
    total_test_items = 0
    history_overlap_items = 0
    users_with_history_overlap = 0
    no_score_users: list[int] = []
    missed_user_examples: list[dict[str, Any]] = []
    overlap_user_examples: list[dict[str, Any]] = []

    for user_id in eval_user_ids:
        positives = test_positives[user_id]
        total_test_items += len(positives)
        if user_id in scorer.user_to_index:
            model_user_hits += 1

        history_items = (
            training_interactions["positives"].get(user_id, set())
            | training_interactions["negatives"].get(user_id, set())
        )
        rec_ids = _bpr_top_k_for_user(
            recommender,
            scorer,
            user_id,
            top_k,
            seen_recipe_ids=history_items if exclude_seen else None,
        )
        if not rec_ids:
            no_score_users.append(user_id)
            continue

        users_with_scores += 1
        returned += len(rec_ids)
        user_hits = set(rec_ids) & positives
        hit_items += len(user_hits)
        if user_hits:
            hit_users += 1
        elif len(missed_user_examples) < 10:
            missed_user_examples.append(
                {
                    "user_id": user_id,
                    "test_positives": sorted(positives),
                    "bpr_top_k": rec_ids,
                }
            )

        overlap = set(rec_ids) & history_items
        history_overlap_items += len(overlap)
        if overlap:
            users_with_history_overlap += 1
            if len(overlap_user_examples) < 10:
                overlap_user_examples.append(
                    {
                        "user_id": user_id,
                        "overlap_items": sorted(overlap),
                        "bpr_top_k": rec_ids,
                    }
                )

    return {
        "skipped": False,
        "exclude_seen": exclude_seen,
        "model_users": len(scorer.user_to_index),
        "model_recipes": len(scorer.recipe_to_index),
        "evaluated_users_in_bpr_model": model_user_hits,
        "evaluated_user_model_coverage": _safe_div(model_user_hits, len(eval_user_ids)),
        "candidate_recipes_in_bpr_model": model_recipe_hits,
        "candidate_recipe_model_coverage": _safe_div(model_recipe_hits, len(recommender.recipes)),
        "users_with_bpr_scores": users_with_scores,
        "users_without_bpr_scores": len(no_score_users),
        "users_without_bpr_score_examples": no_score_users[:10],
        "hit_users": hit_users,
        "hit_user_rate": _safe_div(hit_users, users_with_scores),
        "hit_items": hit_items,
        "recall_at_k": _safe_div(hit_items, total_test_items),
        "precision_at_k": _safe_div(hit_items, returned),
        "history_overlap_items": history_overlap_items,
        "history_overlap_rate": _safe_div(history_overlap_items, returned),
        "users_with_history_overlap": users_with_history_overlap,
        "users_with_history_overlap_rate": _safe_div(
            users_with_history_overlap, users_with_scores
        ),
        "missed_user_examples": missed_user_examples,
        "history_overlap_user_examples": overlap_user_examples,
    }

def _bpr_top_k_for_user(
    recommender: DACHLLMRecommender,
    scorer: BPRScorer,
    user_id: int,
    top_k: int,
    seen_recipe_ids: set[int] | None = None,
) -> list[int]:
    candidate_recipe_ids = [
        recipe_id
        for recipe_id in recommender.recipes
        if recipe_id in scorer.recipe_to_index
    ]
    return scorer.topk(
        user_id=user_id,
        top_k=top_k,
        candidate_recipe_ids=candidate_recipe_ids,
        exclude_recipe_ids=seen_recipe_ids,
    )


def _count_items(items_by_user: dict[int, set[int]]) -> int:
    return sum(len(items) for items in items_by_user.values())


def _negative_sampling_summary(
    train: dict[str, dict[int, set[int]]],
    candidate_recipes: list[int],
    negative_samples_per_positive: int,
) -> dict[str, Any]:
    candidate_set = set(candidate_recipes)
    training_user_ids = sorted(train["positives"])
    usable_user_ids = [
        user_id
        for user_id in training_user_ids
        if train["positives"][user_id] & candidate_set
    ]
    total_positive_items = 0
    explicit_negative_triples = 0
    sampled_negative_triples = 0
    sampled_pool_sizes: list[int] = []
    explicit_negative_users = 0
    for user_id in usable_user_ids:
        positive_items = train["positives"][user_id] & candidate_set
        explicit_negative_items = train["negatives"].get(user_id, set()) & candidate_set
        sampled_pool = candidate_set - positive_items
        if not positive_items or not sampled_pool:
            continue
        total_positive_items += len(positive_items)
        explicit_negative_users += 1 if explicit_negative_items else 0
        sampled_pool_sizes.append(len(sampled_pool))
        explicit_per_positive = min(len(explicit_negative_items), negative_samples_per_positive)
        explicit_negative_triples += len(positive_items) * explicit_per_positive
        sampled_negative_triples += len(positive_items) * negative_samples_per_positive - len(positive_items) * explicit_per_positive
    total_triples = sampled_negative_triples + explicit_negative_triples
    return {
        "negative_samples_per_positive": negative_samples_per_positive,
        "training_users_used": len(usable_user_ids),
        "positive_items_used": total_positive_items,
        "explicit_negative_users": explicit_negative_users,
        "explicit_negative_triples": explicit_negative_triples,
        "random_negative_triples": sampled_negative_triples,
        "random_negative_ratio": _safe_div(sampled_negative_triples, total_triples),
        "avg_sampled_pool_size": _safe_div(sum(sampled_pool_sizes), len(sampled_pool_sizes)),
    }


def _average_count(items_by_user: dict[int, set[int]], users: set[int]) -> float:
    if not users:
        return 0.0
    return sum(len(items_by_user.get(user_id, set())) for user_id in users) / len(users)


def _safe_div(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose BPR train/test coverage.")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite database path")
    parser.add_argument("--cutoff", default="2026-06-01 00:00:00")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=500)
    parser.add_argument("--bpr-model", default=None, help="Optional trained BPR .pt artifact")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args(argv)

    result = diagnose_bpr(
        db_path=args.db,
        cutoff=args.cutoff,
        top_k=args.top_k,
        max_users=args.max_users,
        bpr_model_path=args.bpr_model,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
