# DACH-LLMRec 算法说明

本文档说明当前项目中的推荐算法逻辑，包括使用哪些数据库表、如何处理数据、如何向量化、如何打分、BPR 如何训练、大模型接口如何使用，以及疾病因素如何扩展。

## 1. 任务定义

目标是根据用户多维信息推荐食材和食谱。

输入：

```text
用户基础信息
用户口味偏好
用户健康目标
用户运动/活动信息
用户历史反馈
用户避免食材/避免菜谱
可选疾病或健康风险因素
```

输出：

```text
Top-K 食谱或食材
综合分数
各模块证据分数
命中的推荐因素
推荐解释
```

当前系统不是疾病诊断系统，也不是临床食疗系统。默认只做健康目标约束推荐。

## 2. 使用的数据表

### 2.1 菜谱和食材

```text
norm_recipe_v1
norm_ingredient_v1
norm_recipe_ingredient_v1
```

用途：

- `norm_recipe_v1` 提供菜谱名称、描述、菜系、烹饪方式、口味标签、内容状态、是否可推荐。
- `norm_ingredient_v1` 提供食材名称、类别、营养状态和来源状态。
- `norm_recipe_ingredient_v1` 建立菜谱和食材之间的关系，并区分主料和辅料。

### 2.2 营养可信度

```text
norm_recipe_nutrition_feature_eligibility_v1
```

用途：

- 判断菜谱营养特征是否适合进入模型。
- 当前分为 `standard`、`sensitivity_only`、`exclude_from_nutrition_model`。

### 2.3 用户画像和反馈

```text
norm_synthetic_user_v1
norm_synthetic_user_taste_v1
norm_synthetic_user_health_goal_v1
norm_synthetic_user_sport_v1
norm_synthetic_feedback_event_v1
```

用途：

- 用户基础画像；
- 用户口味；
- 用户健康目标；
- 用户运动习惯；
- 用户隐式反馈。

这些表当前是 synthetic 数据，只能用于模拟用户实验。

### 2.4 健康目标知识

```text
hci
hcirecommendrecipe
hcirecommendingredient
```

用途：

- `hci` 表示健康目标体系；
- `hcirecommendrecipe` 表示健康目标推荐菜谱；
- `hcirecommendingredient` 表示健康目标推荐食材。

### 2.5 口味知识

```text
taste
ingredient2taste
```

用途：

- 建立食材和口味之间的映射；
- 进一步得到菜谱口味向量。

### 2.6 疾病扩展表

```text
diseaseavoidrecipe
diseaseavoidingredient
diseaserecommendrecipe
diseaserecommendingredient
```

用途：

- 后续如果补充可靠 `user -> disease/risk` 数据，可以作为疾病约束和疾病推荐信号。

当前不把 `disease` 表当作用户画像，因为数据库中没有可靠的用户疾病诊断关系。

## 3. 用户向量构造

用户画像表示为：

```text
UserProfile =
  age_years
  sex
  activity_level
  diet_goal
  taste_preferences
  health_goals
  sport_summary
  favored_ingredients
  avoided_ingredients
  avoided_recipes
```

### 3.1 口味向量

用户口味偏好来自：

```text
norm_synthetic_user_taste_v1.preference
```

原始取值：

```text
-2, -1, 0, 1, 2
```

归一化：

```text
taste_weight = preference / 2
```

范围：

```text
-1.0 到 1.0
```

含义：

```text
-1.0 强烈不喜欢
 0.0 中性
 1.0 强烈喜欢
```

### 3.2 健康目标向量

用户健康目标来自：

```text
norm_synthetic_user_health_goal_v1
```

优先级转换为权重：

```text
health_weight = 1 / priority
```

例如：

```text
priority = 1 -> 1.0
priority = 2 -> 0.5
```

### 3.3 HCI 层级扩展

健康目标表 `hci` 有父子层级。为了避免父目标和子目标无法命中规则，系统会扩展健康目标：

```text
父目标 -> 子目标，权重乘 0.8
子目标 -> 父目标，权重乘 0.5
```

例如用户目标是“免疫调节”，规则表中是“增强免疫”，通过层级扩展后仍然可以匹配。

### 3.4 历史反馈

反馈来自：

```text
norm_synthetic_feedback_event_v1
```

事件权重：

```text
cook       =  5.0
save       =  4.0
click      =  2.0
impression =  0.5
skip       = -1.0
dislike    = -4.0
```

## 4. 菜谱和食材特征构造

### 4.1 菜谱-食材向量

根据：

```text
norm_recipe_ingredient_v1
```

构建菜谱食材集合。

权重：

```text
主料 = 1.0
辅料 = 0.5
```

### 4.2 菜谱口味向量

路径：

```text
recipe -> ingredient -> ingredient2taste -> taste
```

对一个菜谱中所有食材的口味进行加权统计，再归一化为口味分布向量。

### 4.3 菜谱健康目标向量

两种来源：

```text
recipe -> hcirecommendrecipe
recipe -> ingredient -> hcirecommendingredient
```

也就是说，菜谱可以直接匹配健康目标，也可以通过它包含的食材间接匹配健康目标。

### 4.4 语义文本向量

用户文本由以下信息拼接：

```text
年龄、性别、活动水平、饮食目标、口味、健康目标、运动、偏好食材
```

菜谱文本由以下信息拼接：

```text
菜谱名称、描述、菜系、做法、口味标签、食材、营养可信度
```

当前默认使用 `HashEmbeddingProvider` 生成离线确定性向量。它不是真实大模型，只是为了保证项目不依赖网络和 API key 也能复现。

## 5. 硬过滤

硬过滤在打分之前执行。

当前规则：

```text
norm_recipe_v1.recommendable 必须等于 1
用户避免菜谱直接排除
菜谱包含用户避免食材直接排除
```

如果启用疾病扩展：

```text
diseaseavoidrecipe 命中的菜谱直接排除
diseaseavoidingredient 命中的食材所在菜谱直接排除
```

这些不是负分，而是直接过滤。原因是安全约束不能被高偏好分抵消。

## 6. 推荐打分公式

默认公式：

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

启用疾病扩展后的公式：

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

疾病扩展默认关闭。

## 7. 各分数如何计算

### 7.1 PreferenceScore

用户口味向量和菜谱口味向量做余弦相似度：

```text
raw = cosine(user_taste_vector, recipe_taste_vector)
PreferenceScore = (raw + 1) / 2
```

最终范围是 `[0, 1]`。

### 7.2 HealthGoalScore

健康目标分由直接命中和间接命中组成：

```text
direct = hcirecommendrecipe 直接命中分
indirect = hcirecommendingredient 食材间接命中平均分

HealthGoalScore = 0.6 * direct + 0.4 * indirect
```

强度归一化：

```text
normalized_intensity = intensity / 5
```

### 7.3 ContentScore

衡量菜谱是否包含用户偏好的食材：

```text
ContentScore =
sum(recipe_ingredient_weight * user_favored_ingredient_weight)
/ sum(recipe_ingredient_weight)
```

如果没有偏好食材证据，使用中性值 `0.5`。

### 7.4 FeedbackScore

如果只有事件聚合：

```text
raw_feedback = sum(event_weight)
FeedbackScore = sigmoid(raw_feedback / 5)
```

如果加载了 BPR 模型：

```text
FeedbackScore = 0.5 * event_score + 0.5 * bpr_score
```

### 7.5 LLMAlignmentScore

用户语义向量和菜谱语义向量做余弦相似度：

```text
LLMAlignmentScore =
(cosine(user_text_embedding, recipe_text_embedding) + 1) / 2
```

当前默认不是大模型 embedding，而是本地哈希向量。后续可替换为真实中文 embedding 模型或 API。

### 7.6 QualityScore

```text
QualityScore = content_status_score * nutrition_tier_score
```

内容完整度：

```text
complete = 1.0
partial  = 0.6
sparse   = 0.3
```

营养可信度：

```text
standard                     = 1.0
sensitivity_only             = 0.6
exclude_from_nutrition_model = 0.2
```

### 7.7 DiversityBoost

Top-K 选择时，系统会降低与已选菜谱过于相似的候选。

相似度考虑：

```text
菜系是否相同
烹饪方式是否重叠
主食材是否重叠
```

计算：

```text
DiversityBoost = 1 - max_similarity_to_selected
```

## 8. BPR 隐式反馈模型

BPR 用于学习用户和菜谱的隐向量。

正反馈：

```text
click
save
cook
```

负反馈：

```text
skip
dislike
```

训练目标：

```text
L_BPR = - mean(log sigmoid(score(u, i+) - score(u, i-)))
```

其中：

```text
i+ 是用户正反馈菜谱
i- 是用户负反馈或采样负例菜谱
```

训练支持：

```text
--device auto
--device cpu
--device cuda
```

如果指定 `--device cuda` 但没有 CUDA，程序会报错，不会假装使用 GPU。

## 9. 大模型如何接入

当前项目没有直接调用大模型 API。原因是为了让 GitHub 项目可复现，不依赖网络和 API key。

但是代码已经预留接口：

```python
class EmbeddingProvider:
    def embed(self, text: str) -> list[float]:
        ...
```

后续可以接：

```text
本地中文 embedding 模型
OpenAI embedding API
其他大模型 embedding 服务
```

接入后，大模型主要承担三件事：

```text
用户画像语义增强
菜谱语义向量生成
推荐解释文本生成
```

大模型不应该直接决定最终推荐结果。最终排序仍由可追溯分数计算得到。

## 10. 疾病因素如何扩展

当前数据库中疾病表能连接少量食材和菜谱，但没有可靠的用户疾病画像。

因此当前主算法使用：

```text
用户 -> HCI 健康目标 -> 食材/菜谱
```

而不是：

```text
用户 -> 疾病诊断 -> 食疗推荐
```

如果后续补充可靠用户疾病/风险表，例如：

```text
user_health_condition(user_id, disease_id, confidence, source, status)
```

就可以启用：

```python
recommend(
    user_id=1,
    health_factors=[{"type": "disease", "id": disease_id}],
    enable_disease_constraints=True,
)
```

启用后：

- `diseaseavoidrecipe` 和 `diseaseavoidingredient` 作为硬过滤；
- `diseaserecommendrecipe` 和 `diseaserecommendingredient` 进入 `DiseaseScore`。

## 11. 实验设计

评估采用时间切分：

```text
训练反馈：event_time < cutoff
测试正样本：event_time >= cutoff 且 event_type in {click, save, cook}
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

指标：

```text
Precision@K
Recall@K
NDCG@K
HitRate@K
Coverage
Diversity
SafetyViolationRate
```

由于当前反馈是 synthetic，实验结果应写成“模拟用户实验结果”。

## 12. 当前局限

当前项目是可复现研究原型，不是最终生产级系统。

主要局限：

```text
用户行为是 synthetic，不是真实用户行为
默认 embedding 不是大模型，只是本地哈希向量
疾病模块缺少可靠 user -> disease 数据
营养数据可信度不完全一致，不能作为精准临床营养依据
BPR 是基础隐式反馈模型，不是完整图神经网络推荐器
```

后续改进方向：

```text
接入真实中文 embedding 模型
在 GPU 上多随机种子训练 BPR
补充真实用户或人工标注反馈
补充可靠用户疾病/风险画像后启用疾病模块
报告消融实验、置信区间和错误分析
```
