from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .embeddings import (
    DEFAULT_EMBEDDING_CACHE_DIR,
    DEFAULT_REAL_EMBEDDING_MODEL,
    build_embedding_provider,
)
from .paths import DEFAULT_DB_PATH
from .recommender import DACHLLMRecommender


def build_llmrec_augmented_edges(
    db_path: str | Path = DEFAULT_DB_PATH,
    output: str | Path | None = None,
    cutoff: str = "2026-06-01 00:00:00",
    top_k: int = 5,
    max_users: int | None = 500,
    min_confidence: float = 0.55,
    embedding_provider: str = "hash",
    embedding_model: str = DEFAULT_REAL_EMBEDDING_MODEL,
    embedding_device: str = "auto",
    embedding_cache_dir: str | Path | None = DEFAULT_EMBEDDING_CACHE_DIR,
) -> dict[str, Any]:
    """Generate LLMRec-style user-recipe augmentation edges.

    This implementation is evidence-constrained and offline: it uses the
    project's current user profile, health-goal, ingredient, feedback, and
    embedding evidence. It does not call an LLM and does not override hard
    recipe safety filters.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")

    db_path = Path(db_path)
    provider = build_embedding_provider(
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_device=embedding_device,
        embedding_cache_dir=embedding_cache_dir,
    )
    recommender = DACHLLMRecommender(
        db_path,
        embedding_provider=provider,
        feedback_before=cutoff,
    )
    try:
        user_ids = _training_user_ids(recommender)
        if max_users is not None:
            user_ids = user_ids[:max_users]

        edges: list[dict[str, Any]] = []
        skipped_users = 0
        for user_id in user_ids:
            try:
                profile = recommender._load_user_profile(user_id)
            except ValueError:
                skipped_users += 1
                continue
            user_embedding = recommender.embedding_provider.embed(recommender._user_text(profile))
            seen_recipe_ids = _seen_recipe_ids(recommender, user_id)
            scored: list[tuple[int, float, dict[str, float]]] = []
            for recipe_id in recommender.recipes:
                if recipe_id in seen_recipe_ids:
                    continue
                if not recommender._passes_recipe_filters(recipe_id, profile, []):
                    continue
                evidence = recommender._recipe_evidence(
                    profile, recipe_id, [], user_embedding, None
                )
                confidence = _augmentation_confidence(evidence)
                if confidence >= min_confidence:
                    scored.append((recipe_id, confidence, evidence))

            scored.sort(key=lambda row: row[1], reverse=True)
            for recipe_id, confidence, evidence in scored[:top_k]:
                edges.append(
                    {
                        "user_id": user_id,
                        "recipe_id": recipe_id,
                        "confidence": round(confidence, 6),
                        "source": "evidence_constrained_llmrec_aug",
                        "reasons": _edge_reasons(evidence),
                        "evidence": {
                            key: round(float(value), 6)
                            for key, value in evidence.items()
                            if key
                            in {
                                "preference_score",
                                "health_goal_score",
                                "content_score",
                                "feedback_score",
                                "llm_alignment_score",
                                "quality_score",
                            }
                        },
                    }
                )

        result = {
            "metadata": {
                "database": str(db_path),
                "cutoff": cutoff,
                "top_k": top_k,
                "max_users": max_users,
                "min_confidence": min_confidence,
                "users_considered": len(user_ids),
                "skipped_users": skipped_users,
                "edge_count": len(edges),
                "boundary": (
                    "Offline LLMRec-style augmentation from structured evidence; "
                    "not real LLM-generated labels and not medical advice."
                ),
            },
            "edges": edges,
        }
    finally:
        recommender.close()

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _training_user_ids(recommender: DACHLLMRecommender) -> list[int]:
    user_ids = {
        user_id
        for user_id, _recipe_id in recommender.recipe_feedback
        if user_id is not None
    }
    return sorted(user_ids)


def _seen_recipe_ids(recommender: DACHLLMRecommender, user_id: int) -> set[int]:
    return {
        recipe_id
        for seen_user_id, recipe_id in recommender.recipe_feedback
        if seen_user_id == user_id
    }


def _augmentation_confidence(evidence: dict[str, float]) -> float:
    return max(
        0.0,
        min(
            1.0,
            0.24 * evidence["preference_score"]
            + 0.24 * evidence["health_goal_score"]
            + 0.18 * evidence["content_score"]
            + 0.14 * evidence["llm_alignment_score"]
            + 0.12 * evidence["quality_score"]
            + 0.08 * evidence["feedback_score"],
        ),
    )


def _edge_reasons(evidence: dict[str, float]) -> list[str]:
    reasons = []
    thresholds = {
        "preference_score": (0.6, "taste_profile_match"),
        "health_goal_score": (0.2, "health_goal_match"),
        "content_score": (0.2, "favored_ingredient_match"),
        "llm_alignment_score": (0.6, "semantic_profile_match"),
        "quality_score": (0.6, "recipe_quality"),
    }
    for key, (threshold, reason) in thresholds.items():
        if evidence.get(key, 0.0) >= threshold:
            reasons.append(reason)
    return reasons


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate LLMRec-style augmentation edges.")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite database path")
    parser.add_argument("--output", default="artifacts/llmrec_aug/augmented_edges.json")
    parser.add_argument("--cutoff", default="2026-06-01 00:00:00")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-users", type=int, default=500)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--embedding-provider", choices=["hash", "real"], default="hash")
    parser.add_argument("--embedding-model", default=DEFAULT_REAL_EMBEDDING_MODEL)
    parser.add_argument("--embedding-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--embedding-cache-dir", default=str(DEFAULT_EMBEDDING_CACHE_DIR))
    args = parser.parse_args(argv)

    result = build_llmrec_augmented_edges(
        db_path=args.db,
        output=args.output,
        cutoff=args.cutoff,
        top_k=args.top_k,
        max_users=args.max_users,
        min_confidence=args.min_confidence,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        embedding_cache_dir=args.embedding_cache_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
