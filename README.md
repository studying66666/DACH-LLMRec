# DACH-LLMRec

DACH-LLMRec 是一个面向食材/食谱推荐的研究型算法项目。它基于 SQLite 膳食数据库，实现了“健康目标约束 + 用户口味/食材偏好 + 隐式反馈 + 语义向量增强 + 可解释推荐”的混合推荐流程。

当前项目已经支持：

- 菜谱推荐和食材推荐；
- 不可推荐菜谱、避免食材、避免菜谱的硬过滤；
- HCI 健康目标推荐；
- 用户口味、食材偏好、历史反馈、内容质量、多样性综合排序；
- PyTorch BPR（Bayesian Personalized Ranking，贝叶斯个性化排序）隐式反馈训练；
  BPR 基于点击、收藏、烹饪、跳过等隐式反馈构造正负样本对，优化用户对正样本的排序分数高于负样本。
- 大模型/embedding 接口预留；
- 疾病因素扩展接口，默认不把疾病表当作用户诊断；
- baseline 和消融实验；
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

完整算法设计、打分公式、实现状态和暂未实现的功能见 [docs/ALGORITHM.md](docs/ALGORITHM.md)。

推荐流程如下：

```text
SQLite 数据库
  -> 读取用户、菜谱、食材、健康目标、口味和反馈
  -> 构建用户画像和菜谱/食材特征
  -> 执行硬过滤
  -> 计算多维推荐分数
  -> 多样性重排
  -> 返回 Top-K 推荐和证据解释
```

菜谱推荐默认打分公式：

```text
Score(u,r) =
0.22 * PreferenceScore(u,r)
+ 0.22 * HealthGoalScore(u,r)
+ 0.16 * ContentScore(u,r)
+ 0.15 * FeedbackScore(u,r)
+ 0.10 * LLMAlignmentScore(u,r)
+ 0.10 * QualityScore(r)
+ 0.05 * DiversityBoost(u,r)
```

如果未来补充了可靠的用户疾病/风险画像，可启用疾病扩展公式：

```text
Score(u,r) =
0.18 * PreferenceScore(u,r)
+ 0.18 * HealthGoalScore(u,r)
+ 0.16 * DiseaseScore(u,r)
+ 0.14 * ContentScore(u,r)
+ 0.12 * FeedbackScore(u,r)
+ 0.10 * LLMAlignmentScore(u,r)
+ 0.08 * QualityScore(r)
+ 0.04 * DiversityBoost(u,r)
```

疾病分数默认不启用，避免把疾病知识表误当成用户诊断信息。

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

## 6. BPR 隐式反馈训练

CPU 烟测：

```bash
python -m dach_llmrec.bpr --db data/demo.sqlite --device cpu --epochs 3 --dim 32 --batch-size 512 --output artifacts/dach_bpr_cpu.pt
```

GPU 训练完整数据库：

```bash
python -m dach_llmrec.bpr --db handoff_database_completed_20260729/dietrecommendation_no_empty_enhanced.sqlite --device cuda --epochs 20 --dim 64 --batch-size 1024 --output artifacts/dach_bpr_gpu.pt --summary-output artifacts/dach_bpr_gpu_summary.json
```

加载 BPR 模型推荐：

```bash
python -m dach_llmrec.cli --db handoff_database_completed_20260729/dietrecommendation_no_empty_enhanced.sqlite --user-id 1 --top-k 5 --mode recipe --bpr-model artifacts/dach_bpr_gpu.pt
```

## 7. 实验评估

单独评估：

```bash
python -m dach_llmrec.evaluate --db handoff_database_completed_20260729/dietrecommendation_no_empty_enhanced.sqlite --cutoff "2026-06-01 00:00:00" --top-k 10 --max-users 500 --bpr-model artifacts/dach_bpr_gpu.pt --output artifacts/dach_bpr_gpu_eval.json
```

一键运行 BPR 训练、baseline 和消融实验：

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
bpr_only
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
  recommender.py     # 主推荐流程
  bpr.py             # PyTorch BPR 训练和加载
  evaluate.py        # 时间切分评估
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

