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

如果需要真实中文 embedding（默认 `BAAI/bge-small-zh-v1.5`），另装可选依赖：

```bash
pip install -e ".[embeddings]"
```

注意 `sentence-transformers` 需要与其 `torch` 版本匹配的 `torchvision`（例如 torch 2.6.0 对应 torchvision 0.21.0）。若导入 `sentence_transformers` 报错，先对齐 torch / torchvision 版本，或继续用默认 `--embedding-provider hash`。

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

DACH 权重网格搜索：

```bash
python -m dach_llmrec.weight_search --db handoff_database_completed_20260729/dietrecommendation_no_empty_enhanced.sqlite --cutoff "2026-06-01 00:00:00" --top-k 10 --max-users 500 --bpr-model artifacts/dach_bpr_gpu.pt --output artifacts/weight_search.json
```

该命令会输出默认权重指标、最优权重、最优验证集指标和排名靠前的候选权重组合。它只改变权重选择，不改变证据公式本身。

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
content_feedback
itemknn
als_only
bpr_only
llmrec_aug_bpr
fusion_lr
dach_grid
dach_no_health
dach_no_llm
dach_no_feedback
dach_no_diversity
dach_full
```

embedding 消融方法（对比语义向量来源对指标的影响，已接入 `run_all`）：

```text
dach_no_semantic      # 关闭 LLM/语义证据项
dach_hash_embedding   # 强制使用 hash provider
dach_real_embedding   # 强制使用 real provider（需 --embedding-provider real，否则跳过并记录原因）
```

所有评估 ranker 共享统一 embedding 参数：`--embedding-provider {hash,real}`、`--embedding-model`、`--embedding-device {auto,cpu,cuda}`、`--embedding-cache-dir`。一键实验传入 `--embedding-provider real` 时，会在 `config.json` / `experiment.json` 写入 `embedding_config`，并单独写出 `embedding_ablation.json`（包含上述三个 embedding 消融 ranker 的指标）。

其中 `itemknn`、`als_only`、`fusion_lr`、`dach_grid` 是优化后新增的离线对比 ranker；`fusion_lr` 会在 cutoff 前 synthetic 反馈上训练逻辑回归融合模型，`dach_grid` 会在验证集上按 `NDCG@K` 搜索 DACH evidence 权重。
注意：当前评估基于 `norm_synthetic_feedback_event_v1`，只能表述为模拟用户实验，不能表述为真实用户验证。

## 6. 测试

```bash
pytest -q
```

测试套件包含 embedding 相关用例：`SentenceTransformerEmbeddingProvider` 的向量返回、磁盘缓存命中、缺失依赖报错提示，以及三个 embedding 消融入口（`dach_no_semantic` / `dach_hash_embedding` / `dach_real_embedding`）的评估与 `run_all` 写入 `embedding_ablation` 的断言。全量 `pytest -q` 已通过（19 passed）；真实 embedding 评估在 `--embedding-provider real` 下不再被跳过，可正常产出指标。

环境修复记录：`sentence-transformers` 之前因 `torchvision` 版本不匹配（旧版 0.2.2）而无法导入；已将 `torchvision` 升级到与 `torch 2.6.0` 匹配的 `0.21.0`，导入恢复正常，真实 embedding smoke 测试已跑通。若导入 `sentence_transformers` 报错，先按第 2 节对齐 `torch` / `torchvision` 版本。

语法检查：

```bash
python -m compileall -q dach_llmrec tests
```

## 7. 大模型 / embedding 接口边界

项目支持两类语义向量 provider，二者实现同一个 `EmbeddingProvider` 接口（`embed(text) -> list[float]`）：

- `HashEmbeddingProvider`（默认）：离线确定性文本向量，不依赖网络或 API key。
- `SentenceTransformerEmbeddingProvider`（真实中文 embedding）：基于 `sentence-transformers`，默认模型 `BAAI/bge-small-zh-v1.5`，支持 `auto`/`cpu`/`cuda` 设备选择，并对编码结果做磁盘缓存（`artifacts/embedding_cache`）。

安装真实 embedding 可选依赖（`torch` 需与 `torchvision` 版本匹配，见第 2 节）：

```bash
pip install -e ".[embeddings]"
```

启用真实 embedding：

```bash
python -m dach_llmrec.cli \
  --db handoff_database_completed_20260729/dietrecommendation_no_empty_enhanced.sqlite \
  --user-id 1 --top-k 5 --mode recipe \
  --embedding-provider real --embedding-model BAAI/bge-small-zh-v1.5 --embedding-device auto
```

或代码内构造：

```python
from dach_llmrec import DACHLLMRecommender, build_embedding_provider

recommender = DACHLLMRecommender(
    db_path,
    embedding_provider=build_embedding_provider("real", model="BAAI/bge-small-zh-v1.5"),
)
```

推荐解释仍然必须基于结构化证据生成，不能编造医学、治疗或治愈结论。
