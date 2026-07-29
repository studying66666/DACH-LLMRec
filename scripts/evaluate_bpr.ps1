$ErrorActionPreference = "Stop"

python -m dach_llmrec.evaluate `
  --db "handoff_database_completed_20260729/dietrecommendation_no_empty_enhanced.sqlite" `
  --cutoff "2026-06-01 00:00:00" `
  --top-k 10 `
  --max-users 500 `
  --bpr-model "artifacts/dach_bpr_gpu.pt" `
  --output "artifacts/dach_bpr_gpu_eval.json"
