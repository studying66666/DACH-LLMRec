# DACH-LLMRec：一种面向膳食推荐的健康目标约束多证据融合排序方法

## 摘要

膳食推荐不仅需要预测用户对食谱和食材的兴趣，还需要同时处理健康目标、禁忌食材、内容质量、营养可信度和推荐解释等约束。针对这一问题，本文基于当前项目代码实现，提出 DACH-LLMRec，一种面向食谱与食材推荐的健康目标约束多证据融合排序方法。系统以 SQLite 膳食数据库为基础，构建用户口味偏好、健康目标、偏好食材、避免项和隐式反馈画像；在排序前执行不可推荐菜谱、用户避免菜谱、用户避免食材以及可选疾病禁忌的硬过滤；在排序阶段融合口味匹配、健康目标匹配、内容偏好、历史反馈、离线语义相似度、内容质量和多样性增益等证据；并通过 PyTorch 实现的 Bayesian Personalized Ranking（BPR）模型从 synthetic 隐式反馈中学习用户-菜谱潜在偏好。当前实现支持食谱推荐、食材推荐、BPR 训练与加载、baseline 对比、消融实验和时间切分离线评估。需要强调的是，当前默认语义向量为确定性哈希向量，并非真实大模型 embedding；当前用户画像和反馈来自 synthetic 表，只能支持模拟实验；疾病因素仅在显式传入疾病 ID 时启用，不能解释为真实用户诊断。基于已运行的 demo 验收，项目 5 个 pytest 用例通过，Python 语法编译通过，demo 规模一键实验能够生成 BPR 模型、评估 JSON 和指标 CSV。本文将该系统定位为一个可复现、可解释、健康目标可扩展的膳食推荐研究原型。

**关键词**：膳食推荐；健康目标约束；混合推荐；隐式反馈；BPR；可解释推荐；多样性重排

## 1. 引言

推荐系统在电商、资讯和多媒体场景中通常以兴趣预测为核心目标，而膳食推荐还涉及更复杂的约束。一个食谱即使符合用户口味，也可能因为不可推荐标记、用户避免食材、营养数据可信度不足或健康目标不匹配而不适合返回。因此，膳食推荐不能只依赖单一协同过滤分数或端到端黑盒排序模型，而需要将硬约束、结构化知识、隐式反馈和可解释证据统一到一个可追溯的排序框架中。

当前项目实现的 DACH-LLMRec 采用“先过滤、后融合、再解释”的设计。系统首先从 SQLite 数据库读取菜谱、食材、菜谱-食材关系、健康目标知识、口味知识、营养可信度、synthetic 用户画像和 synthetic 反馈；然后构建用户画像与候选项特征；接着过滤不可推荐项和用户避免项；最后通过多项证据分数的加权融合返回 Top-K 推荐。与只依赖协同过滤的方法相比，该实现保留了健康目标、食材偏好和内容质量等结构化证据；与纯规则系统相比，该实现又通过隐式反馈权重和 BPR 模型引入了个性化学习信号。

本文贡献限定在当前代码已经实现的范围内：

1. 设计并实现了一个面向食谱和食材双任务的健康目标约束推荐流程。
2. 将不可推荐项、避免菜谱、避免食材和显式疾病禁忌作为排序前硬过滤，避免安全约束被偏好分数抵消。
3. 构建了由口味、健康目标、内容、反馈、语义相似度、质量和多样性组成的多证据加权排序公式。
4. 实现了基于 synthetic 隐式反馈的 BPR 训练、模型保存和推荐阶段分数融合。
5. 实现了 popularity、content、bpr_only 以及多种 DACH 消融版本的时间切分评估流程。
6. 通过模板化解释返回每个推荐项的命中因素和证据字段，避免生成无依据的医学结论。

本文不声称已经完成真实大模型推荐、真实用户实验、临床营养建议、图神经网络推荐或生产级服务，因为这些内容没有在当前代码中实现。

## 2. 相关工作

矩阵分解是推荐系统中的经典方法，Koren 等总结了用户和物品隐向量在推荐系统中的作用，为后续个性化排序模型提供了基础。BPR 面向隐式反馈推荐提出 pairwise 排序优化思想，其目标是使用户对正反馈物品的分数高于负反馈物品。当前项目中的 BPR 模块采用了这一思想：通过用户 embedding 和菜谱 embedding 点积得到偏好分数，并使用正负样本三元组优化排序差异。

在评估方面，NDCG 源于信息检索中的累计增益评价思想，适合衡量命中项在 Top-K 排名中的位置质量。当前项目实现了 Precision@K、Recall@K、NDCG@K、HitRate@K、Coverage、Diversity 和 SafetyViolationRate，以覆盖准确性、覆盖度、多样性和安全约束违反率。

与上述通用推荐方法不同，DACH-LLMRec 的实现重点不在提出新的深度模型结构，而在于把膳食推荐中的结构化健康目标、避免项约束、内容质量和隐式反馈学习结合到一个可解释的原型系统中。

## 3. 数据与系统边界

当前代码读取的核心数据表包括菜谱表 `norm_recipe_v1`、食材表 `norm_ingredient_v1`、菜谱-食材关系表 `norm_recipe_ingredient_v1`、营养可信度表 `norm_recipe_nutrition_feature_eligibility_v1`、synthetic 用户画像与反馈表、健康目标知识表、口味知识表和疾病扩展知识表。用户画像和反馈来自 synthetic 表，因此实验只能称为模拟实验，不能写成真实用户实验。疾病知识表不等于用户诊断表。代码没有读取可靠的 `user -> disease` 关系，疾病约束默认关闭；只有调用方显式传入疾病 ID 并开启 `enable_disease_constraints` 时才使用疾病过滤和疾病推荐分。默认语义向量由本地 `HashEmbeddingProvider` 生成，该向量实现用于离线复现和接口占位，不是真实大模型输出。

## 4. 方法

### 4.1 总体流程

DACH-LLMRec 的推荐流程如下：

```text
SQLite 数据库
  -> 读取菜谱、食材、用户画像、健康目标、口味、反馈和扩展知识
  -> 构建 UserProfile、菜谱特征和食材特征
  -> 执行不可推荐项、避免项和可选疾病禁忌硬过滤
  -> 计算多证据分数
  -> 对菜谱候选进行多样性重排
  -> 返回 Top-K 推荐、证据字段、命中因素和模板化解释
```

### 4.2 用户画像构建

代码中的用户画像定义为：

```text
UserProfile =
  user_id,
  age_years,
  sex,
  activity_level,
  diet_goal,
  taste_preferences,
  health_goals,
  sport_summary,
  favored_ingredients,
  avoided_ingredients,
  avoided_recipes
```

用户口味偏好来自 synthetic 用户口味表 `norm_synthetic_user_taste_v1`。其中 `taste_id` 表示口味编号，`preference` 表示模拟用户 \(u\) 对口味 \(k\) 的离散偏好强度。当前代码和 demo 数据按 \(\{-2,-1,0,1,2\}\) 使用该字段：正值表示偏好，负值表示不偏好，0 表示中性。系统将该字段线性归一化为内部口味权重：

\[
t_u(k)=\frac{preference(u,k)}{2}
\]

因此 \(t_u(k)\in[-1,1]\)。需要说明的是，该字段来自 synthetic 用户画像，不能表述为真实用户问卷或真实用户行为数据。

用户健康目标来自 synthetic 用户健康目标表 `norm_synthetic_user_health_goal_v1`。其中 `hci_id` 表示健康目标编号，`priority` 表示模拟用户 \(u\) 对健康目标 \(c\) 的优先级。当前实现假设 `priority` 数值越小，目标越重要，并将其转换为健康目标权重：

\[
h_u(c)=\frac{1}{\max(priority(u,c),1)}
\]

例如，`priority=1` 时 \(h_u(c)=1.0\)，`priority=2` 时 \(h_u(c)=0.5\)。该字段同样来自 synthetic 用户画像，并且 `hci_id` 对应健康目标知识表 `hci`，不是疾病诊断字段。

健康目标表 `hci` 包含父子层级。代码对用户健康目标做层级扩展：父目标向子目标传播时权重乘 0.8，子目标向父目标传播时权重乘 0.5，并对重复目标保留最大权重。

### 4.3 候选项特征构建

菜谱候选项由结构化字段和关系表共同表示。设 \(r\) 表示一个菜谱，\(i\) 表示一个食材，\(I_r\) 表示菜谱 \(r\) 包含的食材集合。代码从 `norm_recipe_v1` 读取菜谱名称、描述、菜系、烹饪方式、口味标签、内容状态和是否可推荐标记；从 `norm_recipe_ingredient_v1` 读取菜谱与食材的对应关系。

在菜谱-食材关系中，`is_main=1` 表示该食材是主料，`is_main=0` 表示该食材作为辅料或普通配料。系统将主料赋予更高权重，辅料赋予较低权重：

\[
w_{r,i}=
\begin{cases}
1.0, & \text{if } i \text{ is a main ingredient of recipe } r,\\
0.5, & \text{otherwise.}
\end{cases}
\]

也就是说，同一个菜谱中，主料对后续口味、内容和健康目标间接匹配的影响更大；辅料仍参与计算，但贡献减半。如果同一食材在同一菜谱中重复出现，代码保留该食材出现过的最大权重。

菜谱口味向量用于表示菜谱在不同口味维度上的分布。设 \(k\) 表示一个口味标签，`ingredient2taste` 表给出食材 \(i\) 与口味 \(k\) 的对应关系。指示函数 \(\mathbb{1}(k\in taste(i))\) 表示：如果食材 \(i\) 具有口味 \(k\)，取 1，否则取 0。系统先按食材权重累计每个口味的贡献，再除以全部口味贡献总和进行归一化：

\[
v_r(k)=
\frac{\sum_{i\in I_r}w_{r,i}\mathbb{1}(k\in taste(i))}
{\sum_{k'}\sum_{i\in I_r}w_{r,i}\mathbb{1}(k'\in taste(i))}
\]

因此，\(v_r(k)\) 可以理解为“菜谱 \(r\) 的整体口味中，口味 \(k\) 所占的加权比例”。例如，一个菜谱的主料和多个辅料都映射到同一口味时，该口味维度的值会更高；如果菜谱没有可用的食材-口味映射，推荐阶段的口味匹配分回退为中性值 0.5。

菜谱健康目标信号分为直接信号和间接信号。直接信号来自 `hcirecommendrecipe`，表示某个健康目标 \(c\) 与菜谱 \(r\) 本身存在推荐关联；间接信号来自 `hcirecommendingredient`，表示菜谱 \(r\) 包含的某些食材 \(i\) 与健康目标 \(c\) 存在推荐关联。后续计算 `HealthGoalScore` 时，代码将直接信号和间接信号按 0.6 与 0.4 融合。

### 4.4 语义向量接口

当前代码提供 `EmbeddingProvider` 接口。默认实现 `HashEmbeddingProvider` 将文本 token 和字符 n-gram 通过 `blake2b` 哈希映射到固定维度向量，并做 L2 归一化。用户文本由年龄、性别、活动水平、饮食目标、口味、健康目标、运动和偏好食材拼接得到；菜谱文本由名称、描述、菜系、做法、口味、食材和营养可信度拼接得到；食材文本由名称、类别、营养状态和来源状态拼接得到。

语义匹配分计算为：

\[
SemanticScore(u,x)=clip_{[0,1]}\left(\frac{\cos(e_u,e_x)+1}{2}\right)
\]

其中 \(x\) 表示菜谱或食材。代码字段名为 `llm_alignment_score`，但当前默认实现不是 LLM embedding。

### 4.5 硬过滤

菜谱推荐在打分前执行以下过滤：

\[
recommendable(r)=1,\quad r\notin AvoidRecipe_u,\quad I_r\cap AvoidIngredient_u=\varnothing
\]

若显式启用疾病 ID 集合 \(D\)，还要求：

\[
r\notin DiseaseAvoidRecipe_D,\quad I_r\cap DiseaseAvoidIngredient_D=\varnothing
\]

食材推荐会过滤用户避免食材；显式启用疾病约束时，还会过滤疾病避免食材。上述过滤属于硬约束，不进入加权求和。

### 4.6 多证据分数

对菜谱 \(r\)，系统计算以下证据：

```text
PreferenceScore
HealthGoalScore
DiseaseScore
ContentScore
FeedbackScore
SemanticScore
QualityScore
DiversityBoost
```

口味匹配分为：

\[
PreferenceScore(u,r)=\frac{\cos(t_u,v_r)+1}{2}
\]

健康目标匹配由直接命中和食材间接命中组成：

\[
HealthGoalScore(u,r)=clip_{[0,1]}(0.6\cdot direct(u,r)+0.4\cdot indirect(u,r))
\]

内容偏好分衡量菜谱食材与用户偏好食材的重合：

\[
ContentScore(u,r)=
\frac{\sum_{i\in I_r}w_{r,i}\cdot favored_u(i)}
{\sum_{i\in I_r}w_{r,i}}
\]

反馈事件权重为：

```text
cook       =  5.0
save       =  4.0
click      =  2.0
impression =  0.5
skip       = -1.0
dislike    = -4.0
```

事件反馈分为：

\[
EventScore(u,r)=\frac{1}{1+\exp(-raw(u,r)/5)}
\]

如果加载 BPR 模型，则：

\[
FeedbackScore(u,r)=clip_{[0,1]}(0.5\cdot EventScore(u,r)+0.5\cdot BPRScore(u,r))
\]

菜谱质量分为内容完整度和营养可信度的乘积：

\[
QualityScore(r)=ContentStatusScore(r)\cdot NutritionTierScore(r)
\]

多样性增益在 Top-K 重排阶段计算。候选菜谱与已选菜谱的相似度为：

\[
sim(r,s)=0.4\cdot \mathbb{1}(cuisine_r=cuisine_s)
+0.3\cdot J(methods_r,methods_s)
+0.3\cdot J(main_r,main_s)
\]

\[
DiversityBoost(r)=1-\max_{s\in Selected}sim(r,s)
\]

其中 \(J(\cdot)\) 为 Jaccard 相似度。

## 5. 排序模型

默认情况下，菜谱综合分为：

\[
\begin{aligned}
Score(u,r)=&
0.22PreferenceScore(u,r)
+0.22HealthGoalScore(u,r)\\
&+0.16ContentScore(u,r)
+0.15FeedbackScore(u,r)\\
&+0.10SemanticScore(u,r)
+0.10QualityScore(r)\\
&+0.05DiversityBoost(r)
\end{aligned}
\]

显式传入疾病 ID 时，综合分切换为：

\[
\begin{aligned}
Score(u,r)=&
0.18PreferenceScore(u,r)
+0.18HealthGoalScore(u,r)\\
&+0.16DiseaseScore(u,r)
+0.14ContentScore(u,r)\\
&+0.12FeedbackScore(u,r)
+0.10SemanticScore(u,r)\\
&+0.08QualityScore(r)
+0.04DiversityBoost(r)
\end{aligned}
\]

食材默认排序分为：

\[
\begin{aligned}
IngredientScore(u,i)=&
0.25PreferenceScore(u,i)
+0.25HealthGoalScore(u,i)\\
&+0.20ContentScore(u,i)
+0.10FeedbackScore(u,i)\\
&+0.10SemanticScore(u,i)
+0.10QualityScore(i)
\end{aligned}
\]

显式启用疾病 ID 后，食材排序分为：

\[
\begin{aligned}
IngredientScore(u,i)=&
0.18PreferenceScore(u,i)
+0.18HealthGoalScore(u,i)\\
&+0.18DiseaseScore(u,i)
+0.18ContentScore(u,i)\\
&+0.08FeedbackScore(u,i)
+0.10SemanticScore(u,i)\\
&+0.10QualityScore(i)
\end{aligned}
\]

## 6. BPR 隐式反馈学习

当前项目在 `dach_llmrec/bpr.py` 中实现 BPR。模型为用户和菜谱分别学习 embedding：

\[
s(u,i)=p_u^\top q_i
\]

其中 \(p_u\) 为用户向量，\(q_i\) 为菜谱向量。训练样本来自 `cutoff` 之前的 synthetic 反馈。正反馈事件为 `click`、`save`、`cook`；负反馈事件为 `skip`、`dislike`。对每个用户构造三元组 \((u,i^+,i^-)\)，优化目标为：

\[
L_{BPR}=
-\frac{1}{|\mathcal{D}|}
\sum_{(u,i^+,i^-)\in\mathcal{D}}
\log\sigma(s(u,i^+)-s(u,i^-))
\]

代码使用 AdamW 优化，并将模型保存为 `.pt` 文件。保存内容包括用户索引、菜谱索引、用户 embedding、菜谱 embedding、训练元数据和每轮 loss。推荐时，`BPRScorer` 加载模型并输出 sigmoid 后的点积分数，作为 `FeedbackScore` 的一部分。

## 7. 推荐解释

当前解释生成是模板化的，不调用大模型。系统根据证据分数阈值生成命中因素，包括口味、健康目标、偏好食材、历史反馈、语义匹配、内容质量和显式启用时的疾病/风险约束。解释文本只说明推荐项通过安全过滤并匹配了哪些已计算证据，不输出治疗、治愈、诊断或临床建议。

## 8. 实验设计与已核验结果

评估采用时间切分。`cutoff` 之前的反馈用于训练和推荐器反馈加载，`cutoff` 之后的正反馈作为测试正样本：

```text
event_time < cutoff   -> training feedback
event_time >= cutoff and event_type in {click, save, cook} -> test positives
```

默认比较方法包括 `popularity`、`content`、`bpr_only`、`dach_no_health`、`dach_no_llm`、`dach_no_feedback`、`dach_no_diversity` 和 `dach_full`。指标包括 Precision@K、Recall@K、NDCG@K、HitRate@K、Coverage、Diversity 和 SafetyViolationRate。

本节仅报告本次实际运行并得到结果的内容。运行：

```text
pytest -q --basetemp .pytest_tmp
```

结果：

```text
5 passed in 3.45s
```

运行：

```text
python -m compileall -q dach_llmrec tests
```

结果：退出码为 0，未输出错误。

运行 demo 实验：

```text
python -m dach_llmrec.experiments.run_all --demo --output-dir artifacts/paper_demo_experiment --device cpu --bpr-epochs 1 --bpr-dim 8 --bpr-batch-size 8 --top-k 3 --max-users 3
```

该实验使用代码生成的 demo SQLite 数据库，不是完整数据集实验，也不是真实用户实验。BPR 训练摘要为：

```text
users = 2
recipes = 4
triples = 6
device = cpu
loss = 0.6916143298149109
boundary = synthetic feedback only; not real-user validation
```

demo 评估结果如下。

| 方法 | Precision@K | Recall@K | NDCG@K | HitRate@K | Coverage | Diversity | SafetyViolationRate |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6000 | 0.9167 | 0.0000 |
| content | 0.3333 | 1.0000 | 0.7500 | 1.0000 | 0.6000 | 0.8333 | 0.0000 |
| bpr_only | 0.1667 | 0.5000 | 0.2500 | 0.5000 | 0.6000 | 0.4167 | 0.0000 |
| dach_no_health | 0.3333 | 1.0000 | 0.7500 | 1.0000 | 0.6000 | 0.8333 | 0.0000 |
| dach_no_llm | 0.3333 | 1.0000 | 0.8155 | 1.0000 | 0.6000 | 0.8333 | 0.0000 |
| dach_no_feedback | 0.3333 | 1.0000 | 0.8155 | 1.0000 | 0.6000 | 0.8333 | 0.0000 |
| dach_no_diversity | 0.3333 | 1.0000 | 0.8155 | 1.0000 | 0.6000 | 0.8333 | 0.0000 |
| dach_full | 0.3333 | 1.0000 | 0.8155 | 1.0000 | 0.6000 | 0.8333 | 0.0000 |

这些指标只证明 demo 数据路径可运行，不能作为发表论文中的主要有效性证据。若要投稿，需要在完整数据库或真实标注数据上补充正式实验。

## 9. 讨论

DACH-LLMRec 的核心价值在于把膳食推荐中的安全约束和排序证据分离。硬过滤保证不可推荐项和避免项不会进入候选集合；多证据排序则允许口味、健康目标、内容偏好、反馈、语义相似度和质量共同影响排名。这一结构适合健康推荐场景，因为它避免了单一兴趣分数覆盖安全约束的问题。

BPR 模块提供了协同过滤学习能力，但当前实现没有让 BPR 直接替代规则排序，而是将其融合到反馈证据中。这种折中设计保留了可解释性，同时允许历史行为影响排序。

疾病扩展的处理较为保守。代码读取疾病推荐和禁忌知识表，但不默认推断用户疾病。这一边界降低了把数据库知识误用为个人医疗判断的风险。若后续存在可靠的用户疾病或健康风险画像，可以在当前接口上启用疾病约束。

## 10. 局限性与投稿前必须补充的内容

当前实现仍存在明显局限。

1. 真实 LLM embedding 未实现。默认 `HashEmbeddingProvider` 是哈希向量，只能作为离线占位。
2. LLM 解释生成未实现。当前解释是模板化文本。
3. 真实用户实验未实现。用户画像和反馈均来自 synthetic 表。
4. 疾病个性化推荐缺少可靠用户疾病画像，疾病模块默认关闭。
5. 当前 BPR 是基础隐式反馈矩阵分解模型，没有实现 LightGCN、GAT、DeepFM 或 Wide&Deep。
6. 当前没有完整数据库上的多随机种子实验、置信区间、显著性检验和错误分析。
7. 当前系统是命令行和 Python 原型，不是生产级在线推荐服务。

如果目标是正式发表，至少需要补充以下材料：完整数据集统计、正式实验设置、多方法对比、消融实验表、参数敏感性分析、案例分析、显著性检验，以及对健康推荐安全边界的伦理说明。

## 11. 结论

本文基于实际代码实现整理了 DACH-LLMRec，一种面向食谱和食材推荐的健康目标约束多证据融合排序原型。系统实现了结构化硬过滤、多维证据打分、BPR 隐式反馈学习、菜谱多样性重排、模板化解释和时间切分评估。与单一协同过滤或纯规则推荐相比，DACH-LLMRec 的实现更强调健康推荐场景中的安全边界、证据可追溯性和模块可扩展性。当前结果表明系统在 demo 数据上可运行，但尚不足以支撑真实场景有效性结论。后续若补充真实 embedding、完整数据实验和严格统计分析，该原型可进一步发展为面向健康膳食推荐的可解释混合推荐系统。

## 参考文献

[1] Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. BPR: Bayesian Personalized Ranking from Implicit Feedback. Conference on Uncertainty in Artificial Intelligence, 2009, 452-461. https://mlanthology.org/uai/2009/rendle2009uai-bpr/

[2] Koren, Y., Bell, R., & Volinsky, C. Matrix Factorization Techniques for Recommender Systems. Computer, 42(8), 30-37, 2009. DOI: 10.1109/MC.2009.263. https://dblp.org/rec/journals/computer/KorenBV09.html

[3] Järvelin, K., & Kekäläinen, J. Cumulated Gain-based Evaluation of IR Techniques. ACM Transactions on Information Systems, 20(4), 422-446, 2002. DOI: 10.1145/582415.582418. https://researchportal.tuni.fi/en/publications/cumulated-gain-based-evaluation-of-ir-techniques

## 附录：本文依据的代码与验证

本文依据以下已读取或已运行的文件：

```text
README.md
docs/ALGORITHM.md
docs/REPRODUCIBILITY.md
pyproject.toml
dach_llmrec/constants.py
dach_llmrec/models.py
dach_llmrec/embeddings.py
dach_llmrec/recommender.py
dach_llmrec/bpr.py
dach_llmrec/evaluate.py
dach_llmrec/experiments/run_all.py
dach_llmrec/demo_data.py
dach_llmrec/cli.py
dach_llmrec/paths.py
tests/test_dach_llmrec.py
```

已运行并得到结果的命令：

```text
pytest -q --basetemp .pytest_tmp
python -m compileall -q dach_llmrec tests
python -m dach_llmrec.experiments.run_all --demo --output-dir artifacts/paper_demo_experiment --device cpu --bpr-epochs 1 --bpr-dim 8 --bpr-batch-size 8 --top-k 3 --max-users 3
```


