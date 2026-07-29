# DACH-LLMRec

健康因素可扩展的食材/食谱推荐原型。当前版本面向你的 SQLite 数据库实现：

- HCI 健康目标约束推荐
- 用户口味、食材偏好、历史反馈融合排序
- BPR 隐式反馈训练
- 可插拔 LLM/embedding 语义增强接口
- 疾病因素扩展接口，默认不把疾病表当成用户诊断
- 推荐解释和硬过滤验收

## 数据边界

默认读取：

```text
handoff_database_completed_20260729/dietrecommendation_no_empty_enhanced.sqlite
```

如果本项目文件夹位于 `D:\Documents\推荐算法\dach_llmrec_project`，代码会自动向上一级查找 `D:\Documents\推荐算法\handoff_database_completed_20260729\...sqlite`。拷到其他机器时，可以把数据库文件夹放到项目同级目录，或通过 `--db` 显式指定。

当前用户画像和反馈来自 synthetic 表，只能用于模拟实验，不能表述为真实用户实验。

当前疾病表能连接少量食材/菜谱，但数据库没有可靠 `user -> disease` 画像，所以疾病模块默认关闭。只有显式传入疾病 ID 时才作为扩展约束使用。

## 安装

```bash
pip install -e ".[dev]"
```

如果要在 GPU 上训练，请先按老师机器 CUDA 版本安装对应 PyTorch，并确认：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## 快速运行

如果没有完整数据库，先生成一个最小 demo 库：

```bash
python -m dach_llmrec.demo_data --output data/demo.sqlite
```

菜谱推荐：

```bash
python -m dach_llmrec.cli --db data/demo.sqlite --user-id 1 --top-k 5 --mode recipe
```

食材推荐：

```bash
python -m dach_llmrec.cli --db data/demo.sqlite --user-id 1 --top-k 5 --mode ingredient
```

硬过滤验收：

```bash
python -m dach_llmrec.cli --db data/demo.sqlite --user-id 1 --top-k 10 --validate
```

## BPR 训练

CPU 烟测：

```bash
python -m dach_llmrec.bpr --db data/demo.sqlite --device cpu --epochs 3 --dim 32 --batch-size 512 --output artifacts/dach_bpr_cpu.pt
```

GPU 训练：

```bash
python -m dach_llmrec.bpr --db handoff_database_completed_20260729/dietrecommendation_no_empty_enhanced.sqlite --device cuda --epochs 20 --dim 64 --batch-size 1024 --output artifacts/dach_bpr_gpu.pt --summary-output artifacts/dach_bpr_gpu_summary.json
```

加载 BPR 模型推荐：

```bash
python -m dach_llmrec.cli --db handoff_database_completed_20260729/dietrecommendation_no_empty_enhanced.sqlite --user-id 1 --top-k 5 --mode recipe --bpr-model artifacts/dach_bpr_gpu.pt
```

## 评估

```bash
python -m dach_llmrec.evaluate --db handoff_database_completed_20260729/dietrecommendation_no_empty_enhanced.sqlite --cutoff "2026-06-01 00:00:00" --top-k 10 --max-users 500 --bpr-model artifacts/dach_bpr_gpu.pt --output artifacts/dach_bpr_gpu_eval.json
```

一键运行训练和全套 baseline/消融实验：

```bash
python -m dach_llmrec.experiments.run_all --db handoff_database_completed_20260729/dietrecommendation_no_empty_enhanced.sqlite --output-dir artifacts/experiment_001 --device cuda --bpr-epochs 20 --bpr-dim 64 --top-k 10 --max-users 500
```

没有完整数据库时可跑 demo 实验：

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

默认评估项包括：

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

## 测试

```bash
pytest -q
python -m compileall -q dach_llmrec tests
```

## 目录

```text
dach_llmrec/
  recommender.py   # 主推荐器、硬过滤、打分、解释、疾病扩展接口
  bpr.py           # PyTorch BPR 训练与模型加载
  evaluate.py      # synthetic feedback 时间切分评估
  demo_data.py     # 最小 demo SQLite 生成器
  cli.py           # 推荐 CLI
  constants.py     # 权重和打分常量
  embeddings.py    # embedding 接口与离线 HashEmbeddingProvider
  models.py        # Recipe / Ingredient / UserProfile 数据模型
  paths.py         # 默认数据库路径发现
  experiments/     # 一键实验 runner
configs/           # GPU 训练参数参考
scripts/           # Windows/Linux 复现脚本
docs/              # 详细复现说明
tests/             # 基础测试
```

## LLM 接入方式

默认 `HashEmbeddingProvider` 是离线确定性文本向量，不是真实大模型。后续可替换：

```python
class MyEmbeddingProvider:
    def embed(self, text: str) -> list[float]:
        ...
```

然后：

```python
from dach_llmrec import DACHLLMRecommender

recommender = DACHLLMRecommender(
    "handoff_database_completed_20260729/dietrecommendation_no_empty_enhanced.sqlite",
    embedding_provider=MyEmbeddingProvider(),
)
```

推荐解释必须基于已计算证据，不输出治疗、治愈或临床建议。
