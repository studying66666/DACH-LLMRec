from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from dach_llmrec.bpr import train_bpr
from dach_llmrec.demo_data import create_demo_database
from dach_llmrec.diagnostics import diagnose_bpr
from dach_llmrec.embeddings import DEFAULT_EMBEDDING_CACHE_DIR, DEFAULT_REAL_EMBEDDING_MODEL
from dach_llmrec.evaluate import evaluate
from dach_llmrec.llmrec_aug import build_llmrec_augmented_edges
from dach_llmrec.paths import DEFAULT_DB_PATH


EMBEDDING_ABLATION_RANKERS = [
    "dach_no_semantic",
    "dach_hash_embedding",
    "dach_real_embedding",
]


def run_all(
    db_path: str | Path | None,
    output_dir: str | Path,
    cutoff: str = "2026-06-01 00:00:00",
    top_k: int = 10,
    max_users: int | None = 500,
    train_bpr_model: bool = True,
    bpr_epochs: int = 20,
    bpr_dim: int = 64,
    bpr_batch_size: int = 1024,
    device: str = "auto",
    use_demo_data: bool = False,
    train_augmented_bpr_model: bool = True,
    augmentation_top_k: int = 5,
    augmentation_min_confidence: float = 0.30,
    embedding_provider: str = "hash",
    embedding_model: str = DEFAULT_REAL_EMBEDDING_MODEL,
    embedding_device: str = "auto",
    embedding_cache_dir: str | Path | None = DEFAULT_EMBEDDING_CACHE_DIR,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_demo_data:
        db_path = create_demo_database(output_dir / "demo.sqlite")
    else:
        db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH

    bpr_path = output_dir / "dach_bpr.pt"
    bpr_summary = None
    if train_bpr_model:
        bpr_summary = train_bpr(
            db_path=db_path,
            output=bpr_path,
            cutoff=cutoff,
            dim=bpr_dim,
            epochs=bpr_epochs,
            batch_size=bpr_batch_size,
            device=device,
        )

    augmented_edges_path = output_dir / "llmrec_augmented_edges.json"
    augmentation_summary = None
    augmented_bpr_path = output_dir / "dach_bpr_llmrec_aug.pt"
    augmented_bpr_summary = None
    if train_bpr_model and train_augmented_bpr_model:
        augmentation_result = build_llmrec_augmented_edges(
            db_path=db_path,
            output=augmented_edges_path,
            cutoff=cutoff,
            top_k=augmentation_top_k,
            max_users=max_users,
            min_confidence=augmentation_min_confidence,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_device=embedding_device,
            embedding_cache_dir=embedding_cache_dir,
        )
        augmentation_summary = augmentation_result["metadata"]
        augmented_bpr_summary = train_bpr(
            db_path=db_path,
            output=augmented_bpr_path,
            cutoff=cutoff,
            dim=bpr_dim,
            epochs=bpr_epochs,
            batch_size=bpr_batch_size,
            device=device,
            augmented_edges_path=augmented_edges_path,
            augmented_min_confidence=augmentation_min_confidence,
        )

    eval_result = evaluate(
        db_path=db_path,
        cutoff=cutoff,
        top_k=top_k,
        max_users=max_users,
        bpr_model_path=bpr_path if bpr_path.exists() else None,
        augmented_bpr_model_path=augmented_bpr_path if augmented_bpr_path.exists() else None,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_device=embedding_device,
        embedding_cache_dir=embedding_cache_dir,
    )
    diagnostics = diagnose_bpr(
        db_path=db_path,
        cutoff=cutoff,
        top_k=top_k,
        max_users=max_users,
        bpr_model_path=bpr_path if bpr_path.exists() else None,
    )

    weight_search = eval_result.get("models", {}).get("dach_grid")
    embedding_config = {
        "provider": embedding_provider,
        "model": embedding_model,
        "device": embedding_device,
        "cache_dir": str(embedding_cache_dir) if embedding_cache_dir else None,
    }
    embedding_ablation = {
        ranker: eval_result["results"].get(ranker)
        for ranker in EMBEDDING_ABLATION_RANKERS
        if ranker in eval_result["results"]
    }

    config = {
        "db_path": str(db_path),
        "cutoff": cutoff,
        "top_k": top_k,
        "max_users": max_users,
        "train_bpr_model": train_bpr_model,
        "bpr_epochs": bpr_epochs,
        "bpr_dim": bpr_dim,
        "bpr_batch_size": bpr_batch_size,
        "device": device,
        "use_demo_data": use_demo_data,
        "train_augmented_bpr_model": train_augmented_bpr_model,
        "augmentation_top_k": augmentation_top_k,
        "augmentation_min_confidence": augmentation_min_confidence,
        "embedding_config": embedding_config,
    }
    output = {
        "config": config,
        "bpr_summary": bpr_summary,
        "augmentation_summary": augmentation_summary,
        "augmented_bpr_summary": augmented_bpr_summary,
        "embedding_ablation": embedding_ablation,
        "diagnostics": diagnostics,
        "weight_search": weight_search,
        "evaluation": eval_result,
    }
    (output_dir / "experiment.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_metrics_csv(output_dir / "metrics.csv", eval_result["results"])
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if weight_search is not None:
        (output_dir / "weight_search.json").write_text(
            json.dumps(weight_search, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if augmented_bpr_summary is not None:
        (output_dir / "augmented_bpr_summary.json").write_text(
            json.dumps(augmented_bpr_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (output_dir / "embedding_ablation.json").write_text(
        json.dumps(embedding_ablation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def _write_metrics_csv(path: Path, results: dict[str, Any]) -> None:
    metric_names = [
        "precision_at_k",
        "recall_at_k",
        "ndcg_at_k",
        "hit_rate_at_k",
        "coverage",
        "diversity",
        "safety_violation_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ranker", *metric_names, "skipped", "reason"])
        writer.writeheader()
        for ranker, metrics in results.items():
            row = {"ranker": ranker}
            if metrics.get("skipped"):
                row["skipped"] = True
                row["reason"] = metrics.get("reason", "")
            for metric_name in metric_names:
                row[metric_name] = metrics.get(metric_name, "")
            writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run DACH-LLMRec training and evaluation.")
    parser.add_argument("--db", default=None, help="SQLite database path")
    parser.add_argument("--output-dir", default="artifacts/experiment_001")
    parser.add_argument("--cutoff", default="2026-06-01 00:00:00")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=500)
    parser.add_argument("--no-train-bpr", action="store_true")
    parser.add_argument("--bpr-epochs", type=int, default=20)
    parser.add_argument("--bpr-dim", type=int, default=64)
    parser.add_argument("--bpr-batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--demo", action="store_true", help="Use generated demo SQLite data")
    parser.add_argument("--no-train-augmented-bpr", action="store_true")
    parser.add_argument("--augmentation-top-k", type=int, default=5)
    parser.add_argument("--augmentation-min-confidence", type=float, default=0.30)
    parser.add_argument("--embedding-provider", choices=["hash", "real"], default="hash")
    parser.add_argument("--embedding-model", default=DEFAULT_REAL_EMBEDDING_MODEL)
    parser.add_argument("--embedding-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--embedding-cache-dir", default=str(DEFAULT_EMBEDDING_CACHE_DIR))
    args = parser.parse_args(argv)

    result = run_all(
        db_path=args.db,
        output_dir=args.output_dir,
        cutoff=args.cutoff,
        top_k=args.top_k,
        max_users=args.max_users,
        train_bpr_model=not args.no_train_bpr,
        bpr_epochs=args.bpr_epochs,
        bpr_dim=args.bpr_dim,
        bpr_batch_size=args.bpr_batch_size,
        device=args.device,
        use_demo_data=args.demo,
        train_augmented_bpr_model=not args.no_train_augmented_bpr,
        augmentation_top_k=args.augmentation_top_k,
        augmentation_min_confidence=args.augmentation_min_confidence,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        embedding_cache_dir=args.embedding_cache_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())