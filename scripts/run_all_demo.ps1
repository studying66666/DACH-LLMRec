$ErrorActionPreference = "Stop"

python -m dach_llmrec.experiments.run_all `
  --demo `
  --output-dir "artifacts/demo_experiment" `
  --top-k 3 `
  --max-users 3 `
  --bpr-epochs 1 `
  --bpr-dim 8 `
  --bpr-batch-size 8 `
  --device cpu
