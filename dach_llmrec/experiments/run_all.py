from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from dach_llmrec.bpr import train_bpr
from dach_llmrec.demo_data import create_demo_database
from dach_llmrec.diagnostics import diagnose_bpr
from dach_llmrec.evaluate import evaluate
from dach_llmrec.paths import DEFAULT_DB_PATH


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

    eval_result = evaluate(
        db_path=db_path,
        cutoff=cutoff,
        top_k=top_k,
        max_users=max_users,
        bpr_model_path=bpr_path if bpr_path.exists() else None,
    )
    diagnostics = diagnose_bpr(
        db_path=db_path,
        cutoff=cutoff,
        top_k=top_k,
        max_users=max_users,
        bpr_model_path=bpr_path if bpr_path.exists() else None,
    )

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
    }
    output = {
        "config": config,
        "bpr_summary": bpr_summary,
        "diagnostics": diagnostics,
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
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
