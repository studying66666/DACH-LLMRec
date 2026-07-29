# DACH-LLMRec 复现实验说明

本文档说明如何在另一台机器，尤其是有 CUDA GPU 的机器上复现训练、推荐和评估。

## 1. 数据位置

默认读取：

```text
handoff_database_completed_20260729/dietrecommendation_no_empty_enhanced.sqlite
```

代码会先查找项目目录下的这个路径，再查找项目上一级目录下的同名路径。如果数据库放在其他路径，所有命令都可以通过 `--db` 指定。

## 2. 环境安装

建议先创建独立环境：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS：

```bash
source .venv/bin/activate
```

安装项目：

```bash
pip install -e ".[dev]"
```

如果要使用 GPU，请按老师机器的 CUDA 版本安装对应 PyTorch。安装后先验证：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

必须看到：

```text
True
```

否则训练会走 CPU，或者 `--device cuda` 会直接报错。

## 3. 训练 BPR 模型

没有完整数据库时，可以先生成 demo 数据库验证流程：

```bash
python -m dach_llmrec.demo_data --output data/demo.sqlite
python -m dach_llmrec.cli --db data/demo.sqlite --user-id 1 --top-k 3 --mode recipe
python -m dach_llmrec.bpr --db data/demo.sqlite --device cpu --epochs 1 --dim 8 --output artifacts/demo_bpr.pt
```

GPU 训练：

```bash
python -m dach_llmrec.bpr --db handoff_database_completed_20260729/dietrecommendation_no_empty_enhanced.sqlite --device cuda --epochs 20 --dim 64 --batch-size 1024 --output artifacts/dach_bpr_gpu.pt --summary-output artifacts/dach_bpr_gpu_summary.json
```

CPU 小规模验证：

```bash
python -m dach_llmrec.bpr --device cpu --epochs 3 --dim 32 --batch-size 512 --output artifacts/dach_bpr_cpu_smoke.pt
```

训练输出会说明实际使用设备：

```json
{
  "device": "cuda"
}
```

如果显示 `"cpu"`，就不能声称使用了 GPU。

## 4. 推荐

不用 BPR 模型：

```bash
python -m dach_llmrec.cli --db handoff_database_completed_20260729/dietrecommendation_no_empty_enhanced.sqlite --user-id 1 --top-k 5 --mode recipe
```

加载 BPR 模型：

```bash
python -m dach_llmrec.cli --db handoff_database_completed_20260729/dietrecommendation_no_empty_enhanced.sqlite --user-id 1 --top-k 5 --mode recipe --bpr-model artifacts/dach_bpr_gpu.pt
```

食材推荐：

```bash
python -m dach_llmrec.cli --user-id 1 --top-k 5 --mode ingredient --bpr-model artifacts/dach_bpr_gpu.pt
```

## 5. 评估

时间切分模拟评估：

```bash
python -m dach_llmrec.evaluate --cutoff "2026-06-01 00:00:00" --top-k 10 --max-users 500 --bpr-model artifacts/dach_bpr_gpu.pt --output artifacts/dach_bpr_gpu_eval.json
```

一键实验，包括训练 BPR、运行 baseline 和消融实验、保存 `experiment.json`、`metrics.csv`、`config.json`：

```bash
python -m dach_llmrec.experiments.run_all --db handoff_database_completed_20260729/dietrecommendation_no_empty_enhanced.sqlite --output-dir artifacts/experiment_001 --device cuda --bpr-epochs 20 --bpr-dim 64 --top-k 10 --max-users 500
```

demo 数据一键实验：

```bash
python -m dach_llmrec.experiments.run_all --demo --output-dir artifacts/demo_experiment --device cpu --bpr-epochs 1 --bpr-dim 8 --top-k 3 --max-users 3
```

指标包括：

```text
Precision@K
Recall@K
NDCG@K
HitRate@K
Coverage
Diversity
SafetyViolationRate
```

默认比较：

```text
popularity
content
bpr_only
dach_no_health
dach_no_llm
dach_no_feedback
dach_no_diversity
dach_full
```

注意：当前评估基于 `norm_synthetic_feedback_event_v1`，只能表述为模拟用户实验，不能表述为真实用户验证。

## 6. 测试

```bash
pytest -q
```

语法检查：

```bash
python -m compileall -q dach_llmrec tests
```

## 7. 大模型接口边界

当前默认 `HashEmbeddingProvider` 是离线确定性文本向量，不是真实 LLM embedding。后续接真实大模型时，只需要实现同样接口：

```python
class MyEmbeddingProvider:
    def embed(self, text: str) -> list[float]:
        ...
```

然后传给：

```python
DACHLLMRecommender(db_path, embedding_provider=MyEmbeddingProvider())
```

推荐解释仍然必须基于结构化证据生成，不能编造医学、治疗或治愈结论。
