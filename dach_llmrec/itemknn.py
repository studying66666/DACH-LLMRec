from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from .constants import FEEDBACK_WEIGHTS


@dataclass
class ItemKNNScorer:
    recipe_to_index: dict[int, int]
    item_similarity: np.ndarray
    user_positive_weights: dict[int, dict[int, float]]
    user_seen_items: dict[int, set[int]]

    @classmethod
    def from_feedback(
        cls,
        recipe_ids: list[int],
        recipe_feedback: dict[tuple[int, int], float],
    ) -> "ItemKNNScorer":
        recipe_to_index = {recipe_id: idx for idx, recipe_id in enumerate(recipe_ids)}
        user_positive_weights: dict[int, dict[int, float]] = defaultdict(dict)
        user_seen_items: dict[int, set[int]] = defaultdict(set)
        user_ids: set[int] = set()

        for (user_id, recipe_id), raw_weight in recipe_feedback.items():
            user_ids.add(user_id)
            user_seen_items[user_id].add(recipe_id)
            if recipe_id not in recipe_to_index:
                continue
            if raw_weight <= 1.0:
                continue
            user_positive_weights[user_id][recipe_id] = float(raw_weight)

        user_order = sorted(user_ids)
        user_to_index = {user_id: idx for idx, user_id in enumerate(user_order)}
        matrix = np.zeros((len(recipe_ids), len(user_order)), dtype=np.float32)
        for user_id, items in user_positive_weights.items():
            user_index = user_to_index[user_id]
            for recipe_id, weight in items.items():
                recipe_index = recipe_to_index.get(recipe_id)
                if recipe_index is not None:
                    matrix[recipe_index, user_index] = float(weight)

        norms = np.linalg.norm(matrix, axis=1)
        safe_norms = np.where(norms > 0.0, norms, 1.0).astype(np.float32)
        normalized = matrix / safe_norms[:, None]
        item_similarity = normalized @ normalized.T
        np.fill_diagonal(item_similarity, 0.0)

        return cls(
            recipe_to_index=recipe_to_index,
            item_similarity=item_similarity.astype(np.float32, copy=False),
            user_positive_weights=user_positive_weights,
            user_seen_items=user_seen_items,
        )

    def topk(
        self,
        user_id: int,
        top_k: int = 10,
        candidate_recipe_ids: list[int] | None = None,
        exclude_recipe_ids: set[int] | None = None,
    ) -> list[int]:
        seen_items = self.user_seen_items.get(user_id, set())
        if exclude_recipe_ids:
            seen_items = seen_items | exclude_recipe_ids

        positive_weights = self.user_positive_weights.get(user_id)
        if not positive_weights:
            return []

        candidate_recipe_ids = candidate_recipe_ids or list(self.recipe_to_index)
        candidate_indices: list[int] = []
        candidate_recipe_list: list[int] = []
        for recipe_id in candidate_recipe_ids:
            if recipe_id in seen_items:
                continue
            recipe_index = self.recipe_to_index.get(recipe_id)
            if recipe_index is None:
                continue
            candidate_recipe_list.append(recipe_id)
            candidate_indices.append(recipe_index)

        if not candidate_recipe_list:
            return []

        scores = np.zeros(self.item_similarity.shape[0], dtype=np.float32)
        for recipe_id, weight in positive_weights.items():
            recipe_index = self.recipe_to_index.get(recipe_id)
            if recipe_index is None:
                continue
            scores += self.item_similarity[recipe_index] * float(weight)

        candidate_scores = scores[np.asarray(candidate_indices, dtype=np.int64)]
        top_n = min(top_k, candidate_scores.shape[0])
        if top_n <= 0:
            return []
        top_local_indices = np.argpartition(-candidate_scores, top_n - 1)[:top_n]
        top_local_indices = top_local_indices[np.argsort(-candidate_scores[top_local_indices])]
        return [candidate_recipe_list[idx] for idx in top_local_indices.tolist()]
