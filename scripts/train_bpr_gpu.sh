#!/usr/bin/env bash
set -euo pipefail

python -m dach_llmrec.bpr \
  --db "handoff_database_completed_20260729/dietrecommendation_no_empty_enhanced.sqlite" \
  --cutoff "2026-06-01 00:00:00" \
  --dim 64 \
  --epochs 20 \
  --batch-size 1024 \
  --learning-rate 0.01 \
  --seed 42 \
  --device cuda \
  --output "artifacts/dach_bpr_gpu.pt" \
  --summary-output "artifacts/dach_bpr_gpu_summary.json"
