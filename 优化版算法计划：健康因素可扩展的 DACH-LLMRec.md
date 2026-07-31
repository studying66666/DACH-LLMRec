# 优化版算法计划：健康因素可扩展的 DACH-LLMRec

## Summary

目标是设计一个适合当前数据库、后续又能平滑加入疾病因素的食材/食谱推荐算法。当前主模型使用 **HCI 健康目标 + 口味 + 食材/菜谱关系 + 模拟反馈 + 可替换语义向量接口**。当前默认语义向量是本地哈希向量，不是真实 LLM embedding；疾病相关表暂不进入默认主排序，但预留 `HealthFactor` 扩展接口，后续补充用户疾病画像后可小改接入。

当前已核验的数据事实：有可推荐菜谱 4066 条、规范食材 7815 条、菜谱-食材关系 12608 条、模拟用户 500 个、模拟反馈事件 23776 条；用户画像没有可靠疾病字段；疾病表能连接少量食材/菜谱，但没有可靠用户-疾病关系。

## Key Design

* 使用主数据表：

  * 菜谱：`norm_recipe_v1`
  * 食材：`norm_ingredient_v1`
  * 菜谱-食材：`norm_recipe_ingredient_v1`
  * 用户画像：`norm_synthetic_user_v1`
  * 用户口味：`norm_synthetic_user_taste_v1`
  * 用户健康目标：`norm_synthetic_user_health_goal_v1`
  * 用户运动：`norm_synthetic_user_sport_v1`
  * 用户反馈：`norm_synthetic_feedback_event_v1`
  * 健康目标知识：`hci`、`hcirecommendrecipe`、`hcirecommendingredient`
  * 口味知识：`taste`、`ingredient2taste`
  * 营养可信度：`norm_recipe_nutrition_feature_eligibility_v1`
* 当前不把疾病表放入主模型：

  * 不用 `disease`、`diseaseavoidingredient`、`diseaseavoidrecipe`、`diseaserecommendingredient`、`diseaserecommendrecipe` 做主排序。
  * 原因是当前没有可靠 `user -> disease` 画像，且疾病-食材/菜谱关系规模较小。
* 预留疾病扩展：

  * 抽象统一健康因素为 `HealthFactor = HCI目标 + 可选疾病/风险标签`。
  * 后续新增用户疾病画像表后，把疾病模块作为 `DiseaseConstraintScore` 接入，不推翻主算法。

## Algorithm

当前算法主线是“硬过滤 + 多证据加权排序 + 菜谱多样性重排”。它不是端到端深度模型，也不是让大模型直接决定排序。

执行步骤：

1. 数据读取：读取菜谱、食材、菜谱-食材关系、HCI 健康目标、HCI 推荐菜谱、HCI 推荐食材、口味知识、营养可信度、synthetic 用户画像和 synthetic 反馈。
2. 用户画像：把用户口味偏好归一化到 [-1,1]；把健康目标优先级转成权重 H_u(c)=1/priority_u(c)；再按 HCI 父子层级传播权重，父到子乘 0.8，子到父乘 0.5。
3. 候选特征：菜谱食材权重按主料 1.0、辅料 0.5 计算；菜谱口味由“菜谱 -> 食材 -> 口味”统计得到；健康目标信号由直接推荐菜谱和间接推荐食材共同提供。
4. 硬过滤：不可推荐菜谱、用户避免菜谱、包含用户避免食材的菜谱直接排除。后续显式启用疾病 ID 时，疾病禁忌菜谱和疾病禁忌食材也直接排除。
5. 多证据打分：分别计算口味匹配、健康目标匹配、内容偏好、历史反馈、语义匹配、质量分和多样性增益。
6. 综合排序：把各证据分数加权求和，得到候选综合分。
7. 多样性重排：菜谱 Top-K 采用贪心选择；每选出一个菜谱，就根据菜系、做法和主食材计算剩余候选与已选集合的相似度，并更新多样性增益。
8. 解释生成：只根据已计算证据生成模板化解释，不生成诊断、治疗或临床营养建议。

默认菜谱排序公式：

$$
\begin{aligned}
Score(u,r)=&0.22PreferenceScore(u,r)+0.22HealthGoalScore(u,r)\\
&+0.16ContentScore(u,r)+0.15FeedbackScore(u,r)\\
&+0.10SemanticScore(u,r)+0.10QualityScore(r)\\
&+0.05DiversityBoost(r)
\end{aligned}
$$

对应权重：

| 证据项 | 权重 |
| --- | ---: |
| PreferenceScore(u,r) | 0.22 |
| HealthGoalScore(u,r) | 0.22 |
| ContentScore(u,r) | 0.16 |
| FeedbackScore(u,r) | 0.15 |
| SemanticScore(u,r) | 0.10 |
| QualityScore(r) | 0.10 |
| DiversityBoost(r) | 0.05 |

疾病扩展后的菜谱排序公式：

$$
\begin{aligned}
Score(u,r)=&0.18PreferenceScore(u,r)+0.18HealthGoalScore(u,r)\\
&+0.16DiseaseScore(r)+0.14ContentScore(u,r)\\
&+0.12FeedbackScore(u,r)+0.10SemanticScore(u,r)\\
&+0.08QualityScore(r)+0.04DiversityBoost(r)
\end{aligned}
$$

对应权重：

| 证据项 | 权重 |
| --- | ---: |
| PreferenceScore(u,r) | 0.18 |
| HealthGoalScore(u,r) | 0.18 |
| DiseaseScore(r) | 0.16 |
| ContentScore(u,r) | 0.14 |
| FeedbackScore(u,r) | 0.12 |
| SemanticScore(u,r) | 0.10 |
| QualityScore(r) | 0.08 |
| DiversityBoost(r) | 0.04 |

各分数含义：

- PreferenceScore：用户口味向量与菜谱口味向量的余弦相似度，映射到 [0,1]。
- HealthGoalScore：HCI 推荐菜谱直接命中占 60%，菜谱所含食材的 HCI 间接命中占 40%。
- ContentScore：用户偏好食材在菜谱食材中的加权覆盖程度。
- FeedbackScore：行为事件权重经 sigmoid 归一化；加载 BPR 后，与 BPR 个性化分各占 50%。
- SemanticScore：用户文本向量与菜谱文本向量的余弦相似度。当前默认是哈希向量，不是真实 LLM embedding。
- QualityScore：内容完整度分乘以营养可信度分。
- DiversityBoost：1 减去候选菜谱与已选菜谱在菜系、做法和主食材上的最大相似度。
- DiseaseScore：仅显式启用疾病 ID 时使用，由疾病推荐菜谱和疾病推荐食材共同计算；疾病禁忌项在硬过滤阶段直接排除。

## LLM Usage

当前实现没有直接调用大模型 API。代码只保留了可替换的 embedding provider 接口，默认使用本地 HashEmbeddingProvider 生成确定性文本向量，便于无网络、无 API key 的复现。

语义向量在当前算法里只负责一个证据项：

$$
\begin{aligned}
SemanticScore(u,x)=\\frac{\\cos(E_u,E_x)+1}{2}
\end{aligned}
$$

其中 E_u 是用户画像文本向量，E_x 是菜谱或食材文本向量。该分数进入综合排序，但不会单独决定最终推荐。

后续如果接入真实 LLM 或中文 embedding 模型，可以替换三处能力：

1. 用户画像语义向量：年龄、性别、活动水平、饮食目标、口味、健康目标、运动习惯和偏好食材。
2. 菜谱/食材语义向量：名称、描述、食材、做法、口味、营养可信度、食材类别和营养状态。
3. 解释文本润色：只能基于已命中的结构化证据表达推荐原因，不能编造医学结论。

论文级训练可以进一步扩展为多目标损失，但当前代码没有实现这些损失，不能写进已完成实验：

$$
\begin{aligned}
L=L_{BPR}+\\lambda_1L_{semantic}+\\lambda_2L_{health}+\\lambda_3L_{safety}+\\lambda_4L_{reg}
\end{aligned}
$$

其中 $L_{semantic}$ 表示语义对齐约束，$L_{health}$ 表示健康目标排序约束，$L_{safety}$ 表示安全违规惩罚，$L_{reg}$ 表示正则项。当前实际已实现的是 BPR 隐式反馈学习和主排序公式融合。
## Interfaces

推荐接口：

```
recommend(user_id: int, top_k: int = 10, mode: str = "recipe") -> dict
```

返回结构：

```
{
  "user_id": 1,
  "mode": "recipe",
  "items": [
    {
      "item_id": 123,
      "item_type": "recipe",
      "name": "示例菜谱",
      "score": 0.86,
      "evidence": {
        "preference_score": 0.81,
        "health_goal_score": 0.74,
        "content_score": 0.69,
        "feedback_score": 0.58,
        "llm_alignment_score": 0.77,
        "quality_score": 0.90
      },
      "matched_factors": ["口味", "健康目标", "偏好食材"],
      "explanation": "基于已命中特征生成的解释"
    }
  ]
}
```

后续疾病扩展接口：

```
recommend(
    user_id: int,
    top_k: int = 10,
    health_factors: list | None = None,
    enable_disease_constraints: bool = False
) -> dict
```

## Test Plan

* 数据验证：

  * 推荐结果全部满足 `recommendable = 1`。
  * 用户避免食材、避免菜谱命中率为 0。
  * 营养 `exclude_from_nutrition_model` 不参与精准营养排序。
  * 当前版本不把 `disease` 表当作用户疾病画像。
* 离线指标：

  * `Precision@K`
  * `Recall@K`
  * `NDCG@K`
  * `HitRate@K`
  * `Coverage`
  * `Diversity`
  * `SafetyViolationRate`
* 对比实验：

  * Popularity
  * Content-based
  * Matrix Factorization
  * BPR
  * LightGCN
  * DACH-LLMRec without LLM
  * Full DACH-LLMRec
* 消融实验：

  * 去掉健康目标图。
  * 去掉 LLM 语义对齐。
  * 去掉异构反馈权重。
  * 去掉质量分。
  * 去掉多样性重排。
* 疾病扩展实验：

  * 只有在补充可靠 `user -> disease/risk` 数据后启用。
  * 单独报告疾病约束命中率和疾病禁忌违规率。
  * 不把疾病实验和当前 HCI 健康目标实验混写。

## Assumptions

* 当前论文/系统主线定位为“健康目标约束的个性化食材与食谱推荐”，不是疾病诊断或疾病食疗推荐。
* 当前用户行为和用户画像来自 synthetic 数据，只能用于模拟实验。
* 疾病模块作为后续可扩展模块保留，不进入当前主模型结论。
* 大模型用于语义增强和解释生成，不作为不可控的最终排序器。
* 推荐解释必须引用数据库和模型已有证据，不输出治疗、治愈、临床建议类表述。
