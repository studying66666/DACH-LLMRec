from __future__ import annotations

import argparse
import json
import random
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .constants import FEEDBACK_WEIGHTS
from .paths import DEFAULT_DB_PATH


POSITIVE_EVENTS = {"click", "save", "cook"}
NEGATIVE_EVENTS = {"skip", "dislike"}


@dataclass
class BPRScorer:
    user_to_index: dict[int, int]
    recipe_to_index: dict[int, int]
    user_embeddings: torch.Tensor
    recipe_embeddings: torch.Tensor
    user_bias: torch.Tensor
    item_bias: torch.Tensor

    @classmethod
    def load(cls, path: str | Path) -> "BPRScorer":
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        user_embeddings = payload["user_embeddings"].float()
        recipe_embeddings = payload["recipe_embeddings"].float()
        user_bias = payload.get("user_bias")
        if user_bias is None:
            user_bias = torch.zeros(user_embeddings.shape[0], dtype=torch.float32)
        else:
            user_bias = user_bias.float().view(-1)
        item_bias = payload.get("item_bias")
        if item_bias is None:
            item_bias = torch.zeros(recipe_embeddings.shape[0], dtype=torch.float32)
        else:
            item_bias = item_bias.float().view(-1)
        return cls(
            user_to_index={int(k): int(v) for k, v in payload["user_to_index"].items()},
            recipe_to_index={int(k): int(v) for k, v in payload["recipe_to_index"].items()},
            user_embeddings=user_embeddings,
            recipe_embeddings=recipe_embeddings,
            user_bias=user_bias,
            item_bias=item_bias,
        )

    def score(self, user_id: int, recipe_id: int) -> float | None:
        user_index = self.user_to_index.get(user_id)
        recipe_index = self.recipe_to_index.get(recipe_id)
        if user_index is None or recipe_index is None:
            return None
        raw = torch.dot(
            self.user_embeddings[user_index], self.recipe_embeddings[recipe_index]
        ) + self.user_bias[user_index] + self.item_bias[recipe_index]
        return torch.sigmoid(raw).item()

    def score_many(
        self,
        user_id: int,
        recipe_ids: list[int] | None = None,
        exclude_recipe_ids: set[int] | None = None,
    ) -> dict[int, float]:
        user_index = self.user_to_index.get(user_id)
        if user_index is None:
            return {}
        recipe_ids = recipe_ids or list(self.recipe_to_index)
        filtered_recipe_ids: list[int] = []
        candidate_indices: list[int] = []
        for recipe_id in recipe_ids:
            if exclude_recipe_ids and recipe_id in exclude_recipe_ids:
                continue
            recipe_index = self.recipe_to_index.get(recipe_id)
            if recipe_index is None:
                continue
            filtered_recipe_ids.append(recipe_id)
            candidate_indices.append(recipe_index)
        if not filtered_recipe_ids:
            return {}

        device = self.recipe_embeddings.device
        index_tensor = torch.as_tensor(candidate_indices, dtype=torch.long, device=device)
        user_vec = self.user_embeddings[user_index].unsqueeze(0)
        item_vecs = self.recipe_embeddings.index_select(0, index_tensor)
        raw = (item_vecs * user_vec).sum(dim=1)
        raw = raw + self.user_bias[user_index] + self.item_bias.index_select(0, index_tensor)
        scores = torch.sigmoid(raw).tolist()
        return {recipe_id: float(score) for recipe_id, score in zip(filtered_recipe_ids, scores)}

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
        filtered_recipe_ids: list[int] = []
        candidate_indices: list[int] = []
        for recipe_id in candidate_recipe_ids:
            if exclude_recipe_ids and recipe_id in exclude_recipe_ids:
                continue
            recipe_index = self.recipe_to_index.get(recipe_id)
            if recipe_index is None:
                continue
            filtered_recipe_ids.append(recipe_id)
            candidate_indices.append(recipe_index)
        if not filtered_recipe_ids:
            return []

        device = self.recipe_embeddings.device
        index_tensor = torch.as_tensor(candidate_indices, dtype=torch.long, device=device)
        user_vec = self.user_embeddings[user_index].unsqueeze(0)
        item_vecs = self.recipe_embeddings.index_select(0, index_tensor)
        raw = (item_vecs * user_vec).sum(dim=1)
        raw = raw + self.user_bias[user_index] + self.item_bias.index_select(0, index_tensor)
        scores = torch.sigmoid(raw)
        top_n = min(top_k, scores.shape[0])
        top_indices = torch.topk(scores, k=top_n).indices.tolist()
        return [filtered_recipe_ids[idx] for idx in top_indices]


class BPRModel(nn.Module):
    def __init__(self, num_users: int, num_items: int, dim: int) -> None:
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, dim)
        self.item_embeddings = nn.Embedding(num_items, dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        nn.init.normal_(self.user_embeddings.weight, std=0.05)
        nn.init.normal_(self.item_embeddings.weight, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(
        self,
        users: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        user_vec = self.user_embeddings(users)
        pos_vec = self.item_embeddings(positives)
        neg_vec = self.item_embeddings(negatives)
        user_bias = self.user_bias(users).squeeze(-1)
        pos_scores = (user_vec * pos_vec).sum(dim=1) + user_bias + self.item_bias(positives).squeeze(-1)
        neg_scores = (user_vec * neg_vec).sum(dim=1) + user_bias + self.item_bias(negatives).squeeze(-1)
        return pos_scores, neg_scores


def train_bpr(
    db_path: str | Path = DEFAULT_DB_PATH,
    output: str | Path = "artifacts/dach_bpr.pt",
    cutoff: str = "2026-06-01 00:00:00",
    dim: int = 32,
    epochs: int = 8,
    batch_size: int = 512,
    learning_rate: float = 0.01,
    seed: int = 42,
    device: str = "auto",
) -> dict[str, Any]:
    """Train a BPR model from synthetic implicit feedback."""

    random.seed(seed)
    torch.manual_seed(seed)
    selected_device = _select_device(device)
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        interactions = _load_training_interactions(conn, cutoff)
        user_ids = sorted(interactions["positives"])
        recipe_ids = _load_candidate_recipes(conn)
        recipe_id_set = set(recipe_ids)
        user_ids = [
            user_id
            for user_id in user_ids
            if interactions["positives"][user_id] & recipe_id_set
        ]
        user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
        recipe_to_index = {recipe_id: idx for idx, recipe_id in enumerate(recipe_ids)}
        if not user_to_index or not recipe_to_index:
            raise ValueError("No usable BPR training users or recipes found.")

        triples = _build_training_triples(
            user_ids=user_ids,
            recipe_ids=recipe_ids,
            positives=interactions["positives"],
            negatives=interactions["negatives"],
            user_to_index=user_to_index,
            recipe_to_index=recipe_to_index,
            negative_samples_per_positive=2,
        )
        if not triples:
            raise ValueError("No BPR triples generated.")

        model = BPRModel(len(user_to_index), len(recipe_to_index), dim).to(selected_device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

        losses = []
        for _epoch in range(epochs):
            random.shuffle(triples)
            epoch_loss = 0.0
            batch_count = 0
            for start in range(0, len(triples), batch_size):
                batch = triples[start : start + batch_size]
                users = torch.tensor([x[0] for x in batch], dtype=torch.long, device=selected_device)
                positives = torch.tensor([x[1] for x in batch], dtype=torch.long, device=selected_device)
                negatives = torch.tensor([x[2] for x in batch], dtype=torch.long, device=selected_device)
                pos_scores, neg_scores = model(users, positives, negatives)
                loss = -F.logsigmoid(pos_scores - neg_scores).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                batch_count += 1
            losses.append(epoch_loss / max(batch_count, 1))

        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "user_to_index": user_to_index,
            "recipe_to_index": recipe_to_index,
            "user_embeddings": model.user_embeddings.weight.detach().cpu(),
            "recipe_embeddings": model.item_embeddings.weight.detach().cpu(),
            "user_bias": model.user_bias.weight.detach().cpu().view(-1),
            "item_bias": model.item_bias.weight.detach().cpu().view(-1),
            "metadata": {
                "db_path": str(db_path),
                "cutoff": cutoff,
                "dim": dim,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "seed": seed,
                "device": str(selected_device),
                "boundary": "trained from synthetic feedback only",
                "bias_terms": True,
            },
            "losses": losses,
        }
        torch.save(payload, str(output))
        return {
            "output": str(output),
            "users": len(user_to_index),
            "recipes": len(recipe_to_index),
            "triples": len(triples),
            "device": str(selected_device),
            "losses": losses,
            "boundary": "synthetic feedback only; not real-user validation",
        }
    finally:
        conn.close()


def _select_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is False.")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    return torch.device(device)


def _load_training_interactions(conn: sqlite3.Connection, cutoff: str) -> dict[str, dict[int, set[int]]]:
    positives: dict[int, set[int]] = defaultdict(set)
    negatives: dict[int, set[int]] = defaultdict(set)
    rows = conn.execute(
        """
        SELECT user_id, recipe_id, event_type
        FROM norm_synthetic_feedback_event_v1
        WHERE event_time < ?
          AND user_id IS NOT NULL AND recipe_id IS NOT NULL
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


def _build_training_triples(
    user_ids: list[int],
    recipe_ids: list[int],
    positives: dict[int, set[int]],
    negatives: dict[int, set[int]],
    user_to_index: dict[int, int],
    recipe_to_index: dict[int, int],
    negative_samples_per_positive: int,
) -> list[tuple[int, int, int]]:
    all_recipes = set(recipe_ids)
    triples: list[tuple[int, int, int]] = []
    for user_id in user_ids:
        positive_items = positives[user_id] & all_recipes
        explicit_negative_items = negatives[user_id] & all_recipes
        sampled_pool = list(all_recipes - positive_items)
        if not sampled_pool:
            continue
        for positive_id in positive_items:
            negative_candidates = list(explicit_negative_items)
            while len(negative_candidates) < negative_samples_per_positive:
                negative_candidates.append(random.choice(sampled_pool))
            for negative_id in negative_candidates[:negative_samples_per_positive]:
                if negative_id == positive_id:
                    continue
                triples.append(
                    (
                        user_to_index[user_id],
                        recipe_to_index[positive_id],
                        recipe_to_index[negative_id],
                    )
                )
    return triples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a BPR model for DACH-LLMRec.")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite database path")
    parser.add_argument("--output", default="artifacts/dach_bpr.pt", help="Model artifact path")
    parser.add_argument("--cutoff", default="2026-06-01 00:00:00")
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--summary-output", default=None, help="Optional JSON summary path")
    args = parser.parse_args(argv)
    result = train_bpr(
        db_path=args.db,
        output=args.output,
        cutoff=args.cutoff,
        dim=args.dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
    )
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
