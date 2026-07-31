# DACH-LLMRec 原型实现

这是一个读取当前 SQLite 数据库的推荐算法原型，实现了“健康目标约束 + 口味/食材内容匹配 + 合成反馈 + 可替换语义向量接口”的 Top-K 推荐。当前默认语义向量是本地哈希向量，不是真实大模型 embedding。


## 算法说明

当前算法采用“先过滤、再融合、再重排、最后解释”的流程：先过滤不可推荐项、用户避免项和可选疾病禁忌项；再融合口味匹配、健康目标匹配、内容偏好、反馈、语义相似度、质量和多样性证据；菜谱推荐最后做贪心多样性重排。

默认菜谱综合分为：

\[
\begin{aligned}
Score(u,r)=&0.22PreferenceScore(u,r)+0.22HealthGoalScore(u,r)\\
&+0.16ContentScore(u,r)+0.15FeedbackScore(u,r)\\
&+0.10SemanticScore(u,r)+0.10QualityScore(r)\\
&+0.05DiversityBoost(r)
\end{aligned}
\]

其中健康目标分由 HCI 推荐菜谱的直接命中和 HCI 推荐食材的间接命中组成；反馈分由行为权重计算，加载 BPR 后再融合 BPR 个性化分；疾病分数默认关闭，只在显式传入疾病 ID 时启用。详细公式和步骤见 `docs/ALGORITHM.md`。

默认数据库路径：

```text
handoff_database_completed_20260729/dietrecommendation_no_empty_enhanced.sqlite
```

## 使用方式

推荐菜谱：

```bash
python -m dach_llmrec.cli --user-id 1 --top-k 5 --mode recipe
```

推荐食材：

```bash
python -m dach_llmrec.cli --user-id 1 --top-k 5 --mode ingredient
```

训练 BPR 隐式反馈模型：

```bash
python -m dach_llmrec.bpr --epochs 20 --dim 64 --batch-size 1024 --output artifacts/dach_bpr.pt
```

使用训练好的 BPR 模型参与推荐：

```bash
python -m dach_llmrec.cli --user-id 1 --top-k 5 --mode recipe --bpr-model artifacts/dach_bpr.pt
```

基础验收：

```bash
python -m dach_llmrec.cli --user-id 1 --top-k 10 --validate
```

离线模拟评估：

```bash
python -m dach_llmrec.evaluate --top-k 10 --max-users 50 --cutoff "2026-06-01 00:00:00"
```

疾病扩展接口示例。当前数据库没有可靠用户-疾病画像，只有显式传入疾病 ID 时才启用疾病约束：

```bash
python -m dach_llmrec.cli --user-id 1 --top-k 5 --disease-id 22
```

## 当前实现边界

- 使用 `norm_synthetic_user_*` 和 `norm_synthetic_feedback_event_v1`，这些是模拟用户/模拟反馈，不能表述为真实用户行为。
- 疾病表默认不进入主模型；仅在显式传入 `--disease-id` 时作为扩展约束使用。
- 默认 `HashEmbeddingProvider` 是离线确定性文本向量，不是真实大模型 embedding。后续可替换为真实 embedding provider。
- 推荐解释只引用结构化证据，不生成治疗、治愈、临床建议。
- `dach_llmrec.evaluate` 使用 synthetic feedback 的时间切分做模拟验收，不代表真实用户实验结果。
- `dach_llmrec.bpr` 会自动选择 `cuda` 或 `cpu`。当前本机验证环境 `torch.cuda.is_available()` 为 `False`，所以本地训练结果是 CPU 训练；在有 CUDA 的 GPU 机器上同一命令会自动使用 GPU。
