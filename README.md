# DACH-LLMRec

DACH-LLMRec 是一个面向食材/食谱推荐的研究型算法项目。它基于 SQLite 膳食数据库，实现了“健康目标约束 + 用户口味/食材偏好 + 隐式反馈 + 语义向量增强 + 可解释推荐”的默认推荐流程，并提供 BPR、ItemKNN、ALS 和 Logistic Fusion 等优化后排序器用于离线对比评估。

当前项目已经支持：

- 菜谱推荐和食材推荐；
- 不可推荐菜谱、避免食材、避免菜谱的硬过滤；
- HCI 健康目标推荐；
- 用户口味、食材偏好、历史反馈、内容质量、多样性综合排序；
- PyTorch BPR（Bayesian Personalized Ranking）隐式反馈训练，已加入用户偏置、物品偏置和批量打分；
- ItemKNN 物品协同过滤、隐式反馈 ALS、Logistic Regression 证据融合、DACH 权重网格搜索；
- 大模型/embedding 接口预留；
- 疾病因素扩展接口，默认不把疾病表当作用户诊断；
- popularity、content、content_feedback、bpr_only、itemknn、als_only、fusion_lr、dach_grid baseline 和 DACH 消融实验；
- 无完整数据库时的 demo 数据复现。

完整算法说明见：[docs/ALGORITHM.md](docs/ALGORITHM.md)。

优化版算法计划原文见：[优化版算法计划：健康因素可扩展的 DACH-LLMRec](<优化版算法计划：健康因素可扩展的 DACH-LLMRec.md>)。

## 1. 数据边界

完整实验默认读取：

```text
handoff_database_completed_20260729/dietrecommendation_no_empty_enhanced.sqlite
```

如果项目文件夹位于：

```text
D:\Documents\推荐算法\dach_llmrec_project
```

代码会自动向上一级查找：

```text
D:\Documents\推荐算法\handoff_database_completed_20260729\dietrecommendation_no_empty_enhanced.sqlite
```

如果数据库在其他位置，可以通过 `--db` 显式指定。

需要注意：

- 当前用户画像和反馈来自 synthetic 表，只能用于模拟实验，不能写成真实用户实验。
- 数据库有疾病知识表，但没有可靠的 `user -> disease` 用户疾病画像。
- 疾病模块默认关闭，只有显式传入疾病 ID 时才启用。
- 推荐解释不能输出治疗、治愈、诊断或临床建议。

## 2. 算法总体思路

完整算法设计、打分公式、实现状态和暂未实现的功能见 [docs/ALGORITHM.md](docs/ALGORITHM.md)。这里给出入口版说明。

当前系统分为两条路径：

1. 默认推荐路径：`recommend()` 仍采用 DACH 可解释排序，即“硬过滤 -> 多证据打分 -> 固定权重融合 -> 菜谱多样性重排 -> 证据解释”。
2. 离线优化评估路径：`evaluate` 新增 `content_feedback`、`itemknn`、`als_only`、`fusion_lr`、`dach_grid` 等 ranker，用来和 `dach_full` 及消融版本对比。

默认 DACH 菜谱综合分为：

$$
\begin{aligned}
Score(u,r)=&0.22PreferenceScore(u,r)+0.22HealthGoalScore(u,r)\\
&+0.16ContentScore(u,r)+0.15FeedbackScore(u,r)\\
&+0.10SemanticScore(u,r)+0.10QualityScore(r)\\
&+0.05DiversityBoost(r)
\end{aligned}
$$

其中，口味分来自用户口味向量与菜谱口味向量的余弦相似度；健康目标分来自 HCI 推荐菜谱的直接命中和推荐食材的间接命中；内容分衡量菜谱食材是否命中用户偏好食材；反馈分来自行为权重和可选 BPR 模型；语义分来自用户文本向量与菜谱文本向量的余弦相似度；质量分来自内容完整度和营养可信度；多样性分用于 Top-K 重排。

BPR 优化后打分为：

$$
BPRScore(u,r)=\sigma(P_u^TQ_r+b_u+b_r)
$$

其中 $b_u$ 和 $b_r$ 分别是用户偏置和菜谱偏置。推荐阶段会通过 `score_many()` 批量计算 BPR 分数，再与事件反馈各占 50% 融入 `FeedbackScore`。

新增离线排序器：

| ranker | 方法 |
| --- | --- |
| content_feedback | 内容分 80% + 反馈分 20% |
| itemknn | 基于正反馈菜谱共现的物品协同过滤 |
| als_only | 隐式反馈 ALS 矩阵分解 |
| fusion_lr | 逻辑回归学习七个 evidence 分数到正反馈概率的映射 |
| dach_grid | 在验证集上按 NDCG@K 搜索 DACH evidence 权重 |

如果未来补充了可靠的用户疾病/风险画像，或调用方显式传入疾病 ID，可启用疾病扩展公式：

$$
\begin{aligned}
Score(u,r)=&0.18PreferenceScore(u,r)+0.18HealthGoalScore(u,r)\\
&+0.16DiseaseScore(r)+0.14ContentScore(u,r)\\
&+0.12FeedbackScore(u,r)+0.10SemanticScore(u,r)\\
&+0.08QualityScore(r)+0.04DiversityBoost(r)
\end{aligned}
$$

疾病分数默认不启用，避免把疾病知识表误当成用户诊断信息。当前默认语义向量是本地哈希向量，不是真实大模型 embedding。
## 3. 主要使用的数据表

菜谱和食材：

```text
norm_recipe_v1
norm_ingredient_v1
norm_recipe_ingredient_v1
```

营养可信度：

```text
norm_recipe_nutrition_feature_eligibility_v1
```

用户画像和反馈：

```text
norm_synthetic_user_v1
norm_synthetic_user_taste_v1
norm_synthetic_user_health_goal_v1
norm_synthetic_user_sport_v1
norm_synthetic_feedback_event_v1
```

健康目标知识：

```text
hci
hcirecommendrecipe
hcirecommendingredient
```

口味知识：

```text
taste
ingredient2taste
```

疾病扩展表：

```text
diseaseavoidrecipe
diseaseavoidingredient
diseaserecommendrecipe
diseaserecommendingredient
```

当前不把 `disease` 表作为用户画像表使用。

## 4. 安装

```bash
pip install -e ".[dev]"
```

如果要在 GPU 上训练，请先安装和 CUDA 版本匹配的 PyTorch，然后验证：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

如果输出 `False`，说明当前环境没有可用 GPU。

## 5. 没有完整数据库时如何运行

生成一个最小 demo SQLite 数据库：

```bash
python -m dach_llmrec.demo_data --output data/demo.sqlite
```

这个命令会重建 `data/demo.sqlite`，并用口味、健康目标、内容和质量分数驱动的 synthetic 反馈替换旧的手工样本。

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

## 6. BPR 训练与优化排序器

CPU 烟测：

```bash
python -m dach_llmrec.bpr --db data/demo.sqlite --device cpu --epochs 3 --dim 32 --batch-size 512 --output artifacts/dach_bpr_cpu.pt
```

GPU 训练完整数据库：

```bash
python -m dach_llmrec.bpr --db handoff_database_completed_20260729/dietrecommendation_no_empty_enhanced.sqlite --device cuda --epochs 20 --dim 64 --batch-size 1024 --output artifacts/dach_bpr_gpu.pt --summary-output artifacts/dach_bpr_gpu_summary.json
```

加载 BPR 模型推荐。当前 BPR 模型包含用户/物品偏置，并支持批量候选打分：

```bash
python -m dach_llmrec.cli --db handoff_database_completed_20260729/dietrecommendation_no_empty_enhanced.sqlite --user-id 1 --top-k 5 --mode recipe --bpr-model artifacts/dach_bpr_gpu.pt
```

DACH 权重网格搜索会枚举多组 evidence 权重，在验证集上按 `NDCG@K` 选择最优组合，并用 `Recall@K`、`Precision@K` 做并列排序：

```bash
python -m dach_llmrec.weight_search --db handoff_database_completed_20260729/dietrecommendation_no_empty_enhanced.sqlite --cutoff "2026-06-01 00:00:00" --top-k 10 --max-users 500 --bpr-model artifacts/dach_bpr_gpu.pt --output artifacts/weight_search.json
```

## 7. 实验评估

单独评估：

```bash
python -m dach_llmrec.evaluate --db handoff_database_completed_20260729/dietrecommendation_no_empty_enhanced.sqlite --cutoff "2026-06-01 00:00:00" --top-k 10 --max-users 500 --bpr-model artifacts/dach_bpr_gpu.pt --output artifacts/dach_bpr_gpu_eval.json
```

一键运行 BPR 训练、baseline、协同过滤 ranker 和消融实验：

```bash
python -m dach_llmrec.experiments.run_all --db handoff_database_completed_20260729/dietrecommendation_no_empty_enhanced.sqlite --output-dir artifacts/experiment_001 --device cuda --bpr-epochs 20 --bpr-dim 64 --top-k 10 --max-users 500
```

demo 数据一键实验：

```bash
python -m dach_llmrec.experiments.run_all --demo --output-dir artifacts/demo_experiment --device cpu --bpr-epochs 1 --bpr-dim 8 --top-k 3 --max-users 3
```

默认比较方法：

```text
popularity
content
content_feedback
itemknn
als_only
bpr_only
fusion_lr
dach_grid
dach_no_health
dach_no_llm
dach_no_feedback
dach_no_diversity
dach_full
```

评估指标：

```text
Precision@K
Recall@K
NDCG@K
HitRate@K
Coverage
Diversity
SafetyViolationRate
```

## 8. 项目结构

```text
dach_llmrec/
  recommender.py     # 默认 DACH 可解释推荐流程
  bpr.py             # PyTorch BPR 训练、偏置项、批量打分和 top-k
  itemknn.py         # ItemKNN 物品协同过滤
  als.py             # 隐式反馈 ALS
  fusion.py          # Logistic Regression 证据融合
  weight_search.py   # DACH evidence 权重网格搜索和 NDCG@K 调参
  evaluate.py        # 时间切分评估和多 ranker 对比
  diagnostics.py     # BPR 数据和推荐诊断
  demo_data.py       # 最小 demo SQLite 数据库生成
  constants.py       # 权重和常量
  embeddings.py      # embedding 接口和 HashEmbeddingProvider
  models.py          # Recipe / Ingredient / UserProfile 数据模型
  paths.py           # 默认数据库路径发现
  experiments/       # 一键实验 runner
configs/             # 实验参数示例
scripts/             # Windows/Linux 辅助脚本
docs/                # 算法说明和复现文档
tests/               # pytest 测试
```

## 9. 测试

```bash
pytest -q
python -m compileall -q dach_llmrec tests
```

## 10. 大模型接入边界

当前默认的 `HashEmbeddingProvider` 是离线确定性文本向量，不是真实大模型 embedding。它的作用是让项目在没有 API key、没有网络的情况下也能完整运行。

后续可以替换为真实 embedding 模型：

```python
class MyEmbeddingProvider:
    def embed(self, text: str) -> list[float]:
        ...
```

使用方式：

```python
from dach_llmrec import DACHLLMRecommender

recommender = DACHLLMRecommender(
    "handoff_database_completed_20260729/dietrecommendation_no_empty_enhanced.sqlite",
    embedding_provider=MyEmbeddingProvider(),
)
```

推荐解释必须基于已计算证据，不能编造医学结论。
