# 优化版算法计划：健康因素可扩展的 DACH-LLMRec

## Summary

目标是设计一个适合当前数据库、后续又能平滑加入疾病因素的食材/食谱推荐算法。当前主模型使用 **HCI 健康目标 + 口味 + 食材/菜谱关系 + 模拟反馈 + LLM 语义增强**；疾病相关表暂不进入主训练和主排序，但预留 `HealthFactor` 扩展接口，后续补充用户疾病画像后可小改接入。

当前已核验的数据事实：有可推荐菜谱 4066 条、规范食材 7815 条、菜谱-食材关系 12608 条、模拟用户 500 个、模拟反馈事件 23776 条；用户画像没有可靠疾病字段；疾病表能连接少量食材/菜谱，但没有可靠用户-疾病关系。

## Key Design

*   使用主数据表：

    *   菜谱：`norm_recipe_v1`
    *   食材：`norm_ingredient_v1`
    *   菜谱-食材：`norm_recipe_ingredient_v1`
    *   用户画像：`norm_synthetic_user_v1`
    *   用户口味：`norm_synthetic_user_taste_v1`
    *   用户健康目标：`norm_synthetic_user_health_goal_v1`
    *   用户运动：`norm_synthetic_user_sport_v1`
    *   用户反馈：`norm_synthetic_feedback_event_v1`
    *   健康目标知识：`hci`、`hcirecommendrecipe`、`hcirecommendingredient`
    *   口味知识：`taste`、`ingredient2taste`
    *   营养可信度：`norm_recipe_nutrition_feature_eligibility_v1`

*   当前不把疾病表放入主模型：

    *   不用 `disease`、`diseaseavoidingredient`、`diseaseavoidrecipe`、`diseaserecommendingredient`、`diseaserecommendrecipe` 做主排序。
    *   原因是当前没有可靠 `user -> disease` 画像，且疾病-食材/菜谱关系规模较小。

*   预留疾病扩展：

    *   抽象统一健康因素为 `HealthFactor = HCI目标 + 可选疾病/风险标签`。
    *   后续新增用户疾病画像表后，把疾病模块作为 `DiseaseConstraintScore` 接入，不推翻主算法。

## Algorithm

*   先硬过滤：

    *   `norm_recipe_v1.recommendable = 1`
    *   排除用户避免食材、避免菜谱。
    *   排除明显不适合推荐的食材/菜谱。
    *   后续若接入疾病，则 `diseaseavoidrecipe` 和 `diseaseavoidingredient` 进入硬过滤。

*   再多路召回：

    *   口味召回：用户喜欢的味型 → 食材 → 菜谱。
    *   健康目标召回：用户 HCI → 推荐食材/菜谱。
    *   行为召回：点击、收藏、烹饪过的菜谱及相似菜谱。
    *   内容召回：偏好食材、菜系、烹饪方式相似菜谱。
    *   冷启动召回：内容完整、可推荐、营养可信度较高的菜谱。

*   排序公式：

```
Score(u,r) =
0.22 * PreferenceScore(u,r)
+ 0.22 * HealthGoalScore(u,r)
+ 0.16 * ContentScore(u,r)
+ 0.15 * FeedbackScore(u,r)
+ 0.10 * LLMAlignmentScore(u,r)
+ 0.10 * QualityScore(r)
+ 0.05 * DiversityBoost(u,r)
```

*   疾病扩展后的公式：

```
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

*   各分数计算：

    *   `PreferenceScore`：用户口味向量与菜谱口味向量的余弦相似度。
    *   `HealthGoalScore`：`0.6 * hcirecommendrecipe命中分 + 0.4 * hcirecommendingredient命中比例`。
    *   `ContentScore`：用户偏好食材/菜系/烹饪方式与菜谱内容向量的相似度。
    *   `FeedbackScore`：按 `cook=5, save=4, click=2, impression=0.5, skip=-1, dislike=-4` 聚合，再归一化。
    *   `LLMAlignmentScore`：用户语义画像 embedding 与菜谱语义 embedding 的余弦相似度。
    *   `QualityScore`：内容完整度 × 营养可信度。
    *   `DiversityBoost`：降低 Top-K 中重复菜系、重复主食材、重复做法的集中度。

## LLM Usage

*   大模型不直接决定推荐结果，只做三个模块：

    *   用户画像语义增强：把年龄、性别、活动水平、饮食目标、口味、健康目标、运动习惯转成文本画像并生成 embedding。
    *   菜谱语义增强：把菜谱名称、描述、食材、做法、口味、营养可信度转成文本 embedding。
    *   推荐解释生成：只基于已命中的结构化证据生成解释，不编造医学结论。

*   训练时加入语义对齐目标：

```
L =
L_BPR
+ λ1 * L_semantic_alignment
+ λ2 * L_health_margin
+ λ3 * L_safety_penalty
+ λ4 * L_regularization
```

默认：

```
λ1 = 0.2
λ2 = 0.2
λ3 = 1.0
λ4 = 1e-4
```

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

*   数据验证：

    *   推荐结果全部满足 `recommendable = 1`。
    *   用户避免食材、避免菜谱命中率为 0。
    *   营养 `exclude_from_nutrition_model` 不参与精准营养排序。
    *   当前版本不把 `disease` 表当作用户疾病画像。

*   离线指标：

    *   `Precision@K`
    *   `Recall@K`
    *   `NDCG@K`
    *   `HitRate@K`
    *   `Coverage`
    *   `Diversity`
    *   `SafetyViolationRate`

*   对比实验：

    *   Popularity
    *   Content-based
    *   Matrix Factorization
    *   BPR
    *   LightGCN
    *   DACH-LLMRec without LLM
    *   Full DACH-LLMRec

*   消融实验：

    *   去掉健康目标图。
    *   去掉 LLM 语义对齐。
    *   去掉异构反馈权重。
    *   去掉质量分。
    *   去掉多样性重排。

*   疾病扩展实验：

    *   只有在补充可靠 `user -> disease/risk` 数据后启用。
    *   单独报告疾病约束命中率和疾病禁忌违规率。
    *   不把疾病实验和当前 HCI 健康目标实验混写。

## Assumptions

*   当前论文/系统主线定位为“健康目标约束的个性化食材与食谱推荐”，不是疾病诊断或疾病食疗推荐。
*   当前用户行为和用户画像来自 synthetic 数据，只能用于模拟实验。
*   疾病模块作为后续可扩展模块保留，不进入当前主模型结论。
*   大模型用于语义增强和解释生成，不作为不可控的最终排序器。
*   推荐解释必须引用数据库和模型已有证据，不输出治疗、治愈、临床建议类表述。
