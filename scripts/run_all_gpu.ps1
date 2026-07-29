$ErrorActionPreference = "Stop"

python -m dach_llmrec.experiments.run_all `
  --db "handoff_database_completed_20260729/dietrecommendation_no_empty_enhanced.sqlite" `
  --output-dir "artifacts/experiment_gpu" `
  --cutoff "2026-06-01 00:00:00" `
  --top-k 10 `
  --max-users 500 `
  --bpr-epochs 20 `
  --bpr-dim 64 `
  --bpr-batch-size 1024 `
  --device cuda
