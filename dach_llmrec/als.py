from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class ALSScorer:
    user_to_index: dict[int, int]
    recipe_to_index: dict[int, int]
    user_factors: np.ndarray
    recipe_factors: np.ndarray
    item_bias: np.ndarray
    user_seen_items: dict[int, set[int]]

    @classmethod
    def from_feedback(
        cls,
        recipe_ids: list[int],
        recipe_feedback: dict[tuple[int, int], float],
        factors: int = 32,
        iterations: int = 8,
        regularization: float = 0.1,
        alpha: float = 15.0,
        seed: int = 42,
    ) -> "ALSScorer":
        recipe_to_index = {recipe_id: idx for idx, recipe_id in enumerate(recipe_ids)}
        user_ids = sorted({user_id for user_id, recipe_id in recipe_feedback if recipe_id in recipe_to_index})
        user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}

        user_interactions: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
        item_interactions: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
        user_seen_items: dict[int, set[int]] = defaultdict(set)
        observed_user_indices: set[int] = set()
        observed_item_indices: set[int] = set()
        positive_counts = np.zeros(len(recipe_ids), dtype=np.float64)

        for (user_id, recipe_id), raw_weight in recipe_feedback.items():
            user_index = user_to_index.get(user_id)
            recipe_index = recipe_to_index.get(recipe_id)
            if user_index is None or recipe_index is None:
                continue
            weight = float(raw_weight)
            confidence = 1.0 + alpha * abs(weight)
            preference = 1.0 if weight > 0.0 else 0.0
            user_interactions[user_index].append((recipe_index, confidence, preference))
            item_interactions[recipe_index].append((user_index, confidence, preference))
            user_seen_items[user_id].add(recipe_id)
            observed_user_indices.add(user_index)
            observed_item_indices.add(recipe_index)
            if preference > 0.0:
                positive_counts[recipe_index] += 1.0

        rng = np.random.default_rng(seed)
        user_factors = rng.normal(scale=0.01, size=(len(user_ids), factors)).astype(np.float64)
        recipe_factors = rng.normal(scale=0.01, size=(len(recipe_ids), factors)).astype(np.float64)
        eye = np.eye(factors, dtype=np.float64)

        for _ in range(iterations):
            recipe_gram = recipe_factors.T @ recipe_factors
            for user_index in range(len(user_ids)):
                interactions = user_interactions.get(user_index)
                if not interactions:
                    continue
                a = recipe_gram + regularization * eye
                b = np.zeros(factors, dtype=np.float64)
                for recipe_index, confidence, preference in interactions:
                    recipe_vec = recipe_factors[recipe_index]
                    a += (confidence - 1.0) * np.outer(recipe_vec, recipe_vec)
                    if preference > 0.0:
                        b += confidence * recipe_vec
                user_factors[user_index] = np.linalg.solve(a, b)

            user_gram = user_factors.T @ user_factors
            for recipe_index in range(len(recipe_ids)):
                interactions = item_interactions.get(recipe_index)
                if not interactions:
                    continue
                a = user_gram + regularization * eye
                b = np.zeros(factors, dtype=np.float64)
                for user_index, confidence, preference in interactions:
                    user_vec = user_factors[user_index]
                    a += (confidence - 1.0) * np.outer(user_vec, user_vec)
                    if preference > 0.0:
                        b += confidence * user_vec
                recipe_factors[recipe_index] = np.linalg.solve(a, b)

        if positive_counts.max() > 0.0:
            item_bias = np.log1p(positive_counts)
            item_bias = (item_bias / item_bias.max()).astype(np.float64, copy=False)
        else:
            item_bias = np.zeros(len(recipe_ids), dtype=np.float64)

        for user_index in range(len(user_ids)):
            if user_index not in observed_user_indices:
                user_factors[user_index] = 0.0
        for recipe_index in range(len(recipe_ids)):
            if recipe_index not in observed_item_indices:
                recipe_factors[recipe_index] = 0.0

        return cls(
            user_to_index=user_to_index,
            recipe_to_index=recipe_to_index,
            user_factors=user_factors.astype(np.float32, copy=False),
            recipe_factors=recipe_factors.astype(np.float32, copy=False),
            item_bias=item_bias.astype(np.float32, copy=False),
            user_seen_items=user_seen_items,
        )

    def score(self, user_id: int, recipe_id: int) -> float | None:
        user_index = self.user_to_index.get(user_id)
        recipe_index = self.recipe_to_index.get(recipe_id)
        if user_index is None or recipe_index is None:
            return None
        raw = float(np.dot(self.user_factors[user_index], self.recipe_factors[recipe_index]))
        return raw + float(self.item_bias[recipe_index])

    def topk(
        self,
        user_id: int,
        top_k: int = 10,
        candidate_recipe_ids: list[int] | None = None,
        exclude_recipe_ids: set[int] | None = None,
    ) -> list[int]:
        user_index = self.user_to_index.get(user_id)
        if user_index is None:
            return []
        candidate_recipe_ids = candidate_recipe_ids or list(self.recipe_to_index)
        seen_items = set(self.user_seen_items.get(user_id, set()))
        if exclude_recipe_ids:
            seen_items |= exclude_recipe_ids

        filtered_recipe_ids: list[int] = []
        candidate_indices: list[int] = []
        for recipe_id in candidate_recipe_ids:
            if recipe_id in seen_items:
                continue
            recipe_index = self.recipe_to_index.get(recipe_id)
            if recipe_index is None:
                continue
            filtered_recipe_ids.append(recipe_id)
            candidate_indices.append(recipe_index)

        if not filtered_recipe_ids:
            return []

        candidate_index_array = np.asarray(candidate_indices, dtype=np.int64)
        user_vec = self.user_factors[user_index]
        item_vecs = self.recipe_factors[candidate_index_array]
        scores = item_vecs @ user_vec + self.item_bias[candidate_index_array]
        top_n = min(top_k, scores.shape[0])
        if top_n <= 0:
            return []
        top_local_indices = np.argpartition(-scores, top_n - 1)[:top_n]
        top_local_indices = top_local_indices[np.argsort(-scores[top_local_indices])]
        return [filtered_recipe_ids[idx] for idx in top_local_indices.tolist()]
