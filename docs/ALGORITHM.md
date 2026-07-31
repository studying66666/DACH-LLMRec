# DACH-LLMRec 优化后算法说明

本文档说明当前代码实际实现的优化版推荐算法。公式使用 GitHub Markdown 支持较好的 `$$ ... $$` 数学块。

当前系统是一个健康目标约束的混合推荐原型。主线上仍采用“硬过滤 + 多证据可解释排序 + 多样性重排”，同时新增了多种协同过滤和学习型融合评估器，用来验证优化后的排序效果。

需要先明确边界：

1. 当前用户画像和反馈来自 synthetic 表，只能称为模拟实验。
2. 默认语义向量由 `HashEmbeddingProvider` 生成，是本地确定性哈希向量，不是真实大模型 embedding。
3. 疾病表默认不代表用户诊断；只有显式传入疾病 ID 并启用疾病约束时，疾病过滤和疾病分数才进入流程。
4. 新增的 ALS、ItemKNN、Fusion LR 和 DACH 权重网格搜索主要用于离线评估和对比排序，不会改变 CLI 默认 `recommend()` 的可解释 DACH 排序路径。

## 1. 优化后总体结构

系统现在包含两层算法。

默认可解释推荐路径：读取数据，构建用户画像和候选项特征，执行安全硬过滤，计算多项证据分，用固定权重融合成综合分，再对菜谱 Top-K 做多样性重排，最后输出证据解释。

离线优化与对比评估路径：在时间切分实验中新增 `content_feedback`、`itemknn`、`als_only`、`fusion_lr` 和 `dach_grid`，并保留 `popularity`、`content`、`bpr_only`、DACH 完整版和 DACH 消融版本。

当前实际用到的方法包括：基于规则的硬过滤、口味向量相似度、HCI 健康目标知识匹配、用户偏好食材内容匹配、隐式反馈事件权重、BPR 矩阵分解、ItemKNN 物品协同过滤、ALS 隐式反馈矩阵分解、Logistic Regression 证据融合、DACH evidence 权重网格搜索、文本向量语义匹配、内容质量评分和多样性重排。

## 2. 核心符号

设用户为 $u$，菜谱为 $r$，食材为 $i$，健康目标为 $c$，疾病或健康风险因素为 $d$。

| 符号 | 含义 |
| --- | --- |
| $T_u$ | 用户口味偏好向量 |
| $T_r$ | 菜谱口味分布向量 |
| $H_u$ | 用户健康目标权重 |
| $I_r$ | 菜谱 $r$ 包含的食材集合 |
| $F_u$ | 用户偏好食材权重 |
| $A_u^r$ | 用户避免菜谱集合 |
| $A_u^i$ | 用户避免食材集合 |
| $E_u$ | 用户文本画像向量 |
| $E_r$ | 菜谱文本向量 |
| $E_i$ | 食材文本向量 |

所有证据分数最终归一化到 $[0,1]$。

## 3. 用户画像构建

### 3.1 口味偏好

用户口味偏好来自 `norm_synthetic_user_taste_v1`。原始偏好为 $-2,-1,0,1,2$，归一化为：

$$
T_u(t)=\frac{p_u(t)}{2}
$$

归一化后 $T_u(t)\in[-1,1]$。

### 3.2 健康目标权重

用户健康目标来自 `norm_synthetic_user_health_goal_v1`。优先级越小，权重越大：

$$
H_u(c)=\frac{1}{priority_u(c)}
$$

HCI 健康目标存在父子层级。系统对用户健康目标做双向扩展：

$$
H_u(c_{child})=\max(H_u(c_{child}),0.8H_u(c_{parent}))
$$

$$
H_u(c_{parent})=\max(H_u(c_{parent}),0.5H_u(c_{child}))
$$

父目标向子目标传播保留 80% 权重，子目标向父目标传播保留 50% 权重；重复命中时取最大值。

### 3.3 反馈事件权重

反馈来自 `norm_synthetic_feedback_event_v1`。事件权重如下：

| 事件 | 权重 |
| --- | ---: |
| cook | 5.0 |
| save | 4.0 |
| click | 2.0 |
| impression | 0.5 |
| skip | -1.0 |
| dislike | -4.0 |

用户 $u$ 对菜谱 $r$ 的聚合反馈为：

$$
RawFeedback(u,r)=\sum_{e\in Events(u,r)}w_e
$$

## 4. 候选特征构建

### 4.1 菜谱食材权重

菜谱-食材关系来自 `norm_recipe_ingredient_v1`。主料和辅料权重不同：

$$
W(r,i)=
\begin{cases}
1.0, & i\text{ 是主料} \\
0.5, & i\text{ 是辅料}
\end{cases}
$$

如果同一食材重复出现，保留最大权重。优化后代码额外维护 `recipe_ingredient_sets`，用于硬过滤、健康目标间接匹配和食材反馈聚合，避免反复从字典键集合临时构造。

### 4.2 菜谱口味向量

菜谱口味由“菜谱 -> 食材 -> 口味”的知识链路聚合：

$$
T_r(t)=\frac{\sum_{i\in I_r}W(r,i)\cdot \mathbb{1}(i\text{ 具有口味 }t)}{\sum_{i\in I_r}W(r,i)}
$$

主料影响更大，辅料贡献减半。

### 4.3 健康目标信号

菜谱健康目标信号分为直接信号和间接信号：

$$
DirectHealth(r,c)=\frac{Intensity(r,c)}{5}
$$

$$
IngredientHealth(i,c)=\frac{Intensity(i,c)}{5}
$$

直接信号来自 `hcirecommendrecipe`，间接信号来自 `hcirecommendingredient`。

## 5. 硬过滤

硬过滤发生在排序之前。被过滤的候选不会进入排序，因此安全约束不会被高兴趣分抵消。

菜谱 $r$ 必须满足：

$$
recommendable(r)=1,\quad r\notin A_u^r,\quad I_r\cap A_u^i=\varnothing
$$

如果显式启用疾病因素 $d$，还必须满足：

$$
r\notin AvoidRecipe(d),\quad I_r\cap AvoidIngredient(d)=\varnothing
$$

食材推荐同样会排除用户避免食材；启用疾病约束时，还会排除疾病禁忌食材。

## 6. 可解释 DACH 证据分

### 6.1 口味匹配分

菜谱口味分使用余弦相似度：

$$
PreferenceScore(u,r)=\frac{\cos(T_u,T_r)+1}{2}
$$

食材口味分取食材关联口味上的用户偏好平均值：

$$
PreferenceScore(u,i)=\frac{1}{2}+\frac{1}{2|T_i|}\sum_{t\in T_i}T_u(t)
$$

缺少口味证据时取中性分 0.5。

### 6.2 健康目标分

菜谱健康目标分由直接匹配和食材间接匹配组成：

$$
Direct(u,r)=\max_{c\in H_u}H_u(c)\cdot DirectHealth(r,c)
$$

$$
Indirect(u,r)=\frac{1}{|I_r|}\sum_{i\in I_r}\max_{c\in H_u}H_u(c)\cdot IngredientHealth(i,c)
$$

$$
HealthGoalScore(u,r)=clip_{[0,1]}\left(0.6Direct(u,r)+0.4Indirect(u,r)\right)
$$

食材健康目标分为：

$$
HealthGoalScore(u,i)=clip_{[0,1]}\left(\max_{c\in H_u}H_u(c)\cdot IngredientHealth(i,c)\right)
$$

### 6.3 内容偏好分

菜谱内容分衡量菜谱食材是否命中用户偏好食材：

$$
ContentScore(u,r)=\frac{\sum_{i\in I_r}W(r,i)\cdot F_u(i)}{\sum_{i\in I_r}W(r,i)}
$$

食材内容分直接使用用户对该食材的偏好权重：

$$
ContentScore(u,i)=F_u(i)
$$

### 6.4 反馈分

事件反馈先经过 sigmoid 归一化：

$$
EventScore(u,r)=\sigma\left(\frac{RawFeedback(u,r)}{5}\right),\quad \sigma(x)=\frac{1}{1+e^{-x}}
$$

如果没有加载 BPR 模型：

$$
FeedbackScore(u,r)=EventScore(u,r)
$$

如果加载了 BPR 模型：

$$
BPRScore(u,r)=\sigma(P_u^TQ_r+b_u+b_r)
$$

$$
FeedbackScore(u,r)=clip_{[0,1]}\left(0.5EventScore(u,r)+0.5BPRScore(u,r)\right)
$$

优化后 BPR 加入了用户偏置 $b_u$ 和物品偏置 $b_r$，并提供 `score_many()` 批量打分，推荐阶段不再逐菜谱调用单点打分。

食材反馈分优化为预聚合：先把用户对包含该食材的菜谱反馈分聚合，再在食材推荐时直接查表：

$$
FeedbackScore(u,i)=\frac{1}{|\mathcal{R}(i)|}\sum_{r\in\mathcal{R}(i)}FeedbackScore(u,r)
$$

其中 $\mathcal{R}(i)$ 表示包含食材 $i$ 且用户有反馈记录的菜谱集合。

### 6.5 语义匹配分

用户画像文本、菜谱文本和食材文本通过 embedding provider 转成向量：

$$
SemanticScore(u,x)=\frac{\cos(E_u,E_x)+1}{2}
$$

其中 $x$ 可以是菜谱 $r$ 或食材 $i$。当前默认 provider 是本地哈希向量，不是真实 LLM embedding。

### 6.6 质量分

菜谱质量分：

$$
QualityScore(r)=ContentStatusScore(r)\cdot NutritionTierScore(r)
$$

| 内容状态 | 分数 |
| --- | ---: |
| complete | 1.0 |
| partial | 0.6 |
| sparse | 0.3 |
| 其他或缺失 | 0.5 |

| 营养可信度 | 分数 |
| --- | ---: |
| standard | 1.0 |
| sensitivity_only | 0.6 |
| exclude_from_nutrition_model | 0.2 |
| 其他或缺失 | 0.5 |

食材质量分：营养状态为 `observed` 时取 1.0，否则取 0.6。

### 6.7 疾病扩展分

疾病分数默认关闭。显式启用疾病 ID 后：

$$
DirectDisease(r)=\max_d RecommendRecipe(d,r)
$$

$$
IndirectDisease(r)=\frac{1}{|I_r|}\sum_{i\in I_r}\max_d RecommendIngredient(d,i)
$$

$$
DiseaseScore(r)=clip_{[0,1]}\left(0.6DirectDisease(r)+0.4IndirectDisease(r)\right)
$$

食材疾病分：

$$
DiseaseScore(i)=\max_d RecommendIngredient(d,i)
$$

疾病禁忌项在硬过滤阶段直接排除，疾病推荐项才进入 `DiseaseScore`。

### 6.8 多样性增益

菜谱 Top-K 使用贪心多样性重排。两个菜谱的相似度为：

$$
sim(r,s)=0.4C(r,s)+0.3J(M_r,M_s)+0.3J(G_r,G_s)
$$

其中 $(Cr,s)$ 表示菜系是否相同，$M_r$ 是做法集合，$G_r$ 是主食材集合， $J$ 是 Jaccard 相似度。

$$
DiversityBoost(r)=1-\max_{s\in Selected}sim(r,s)
$$

优化后多样性重排只在候选池前 `max(200, 25K)` 个项目中贪心选择，降低全量重排成本。

## 7. DACH 综合排序公式

### 7.1 默认菜谱排序

$$
\begin{aligned}
Score(u,r)=&0.22PreferenceScore(u,r)+0.22HealthGoalScore(u,r)\\
&+0.16ContentScore(u,r)+0.15FeedbackScore(u,r)\\
&+0.10SemanticScore(u,r)+0.10QualityScore(r)\\
&+0.05DiversityBoost(r)
\end{aligned}
$$

### 7.2 疾病扩展菜谱排序

$$
\begin{aligned}
Score(u,r)=&0.18PreferenceScore(u,r)+0.18HealthGoalScore(u,r)\\
&+0.16DiseaseScore(r)+0.14ContentScore(u,r)\\
&+0.12FeedbackScore(u,r)+0.10SemanticScore(u,r)\\
&+0.08QualityScore(r)+0.04DiversityBoost(r)
\end{aligned}
$$

### 7.3 默认食材排序

$$
\begin{aligned}
IngredientScore(u,i)=&0.25PreferenceScore(u,i)+0.25HealthGoalScore(u,i)\\
&+0.20ContentScore(u,i)+0.10FeedbackScore(u,i)\\
&+0.10SemanticScore(u,i)+0.10QualityScore(i)
\end{aligned}
$$

### 7.4 疾病扩展食材排序

$$
\begin{aligned}
IngredientScore(u,i)=&0.18PreferenceScore(u,i)+0.18HealthGoalScore(u,i)\\
&+0.18DiseaseScore(i)+0.18ContentScore(u,i)\\
&+0.08FeedbackScore(u,i)+0.10SemanticScore(u,i)\\
&+0.10QualityScore(i)
\end{aligned}
$$

## 8. 新增协同过滤与学习型融合

### 8.1 BPR 优化

BPR 训练样本为 $(u,r^+,r^-)$，表示用户 $u$ 对正样本菜谱 $r^+$ 的偏好高于负样本 $r^-$。

$$
\hat{y}_{u,r}=P_u^TQ_r+b_u+b_r
$$

$$
L_{BPR}=-\frac{1}{|\mathcal{D}|}\sum_{(u,r^+,r^-)\in\mathcal{D}}\log\sigma(\hat{y}_{u,r^+}-\hat{y}_{u,r^-})
$$

相比旧版，当前 BPR 增加用户偏置和物品偏置，并支持 `score_many()` 和 `topk()` 批量候选打分。

### 8.2 ItemKNN

ItemKNN 使用训练期正向反馈构造“菜谱 × 用户”矩阵。只有权重大于 1.0 的反馈进入正向权重。

$$
sim_{item}(r_a,r_b)=\frac{X_{r_a}\cdot X_{r_b}}{\|X_{r_a}\|\|X_{r_b}\|}
$$

用户 $u$ 对候选菜谱 $r$ 的 ItemKNN 分数为：

$$
ItemKNNScore(u,r)=\sum_{h\in H_u^+}sim_{item}(r,h)\cdot RawFeedback(u,h)
$$

其中 $H_u^+$ 是用户训练期正反馈菜谱集合。推荐时排除用户已交互菜谱，并仍执行 DACH 的硬过滤。

### 8.3 隐式反馈 ALS

ALS 把聚合反馈权重转成隐式偏好和置信度：

$$
p_{u,r}=\begin{cases}
1, & RawFeedback(u,r)>0\\
0, & RawFeedback(u,r)\le 0
\end{cases}
$$

$$
c_{u,r}=1+\alpha |RawFeedback(u,r)|
$$

ALS 交替更新用户因子 $P_u$ 和菜谱因子 $Q_r$：

$$
P_u=(Q^TC_uQ+\lambda I)^{-1}Q^TC_up_u
$$

$$
Q_r=(P^TC_rP+\lambda I)^{-1}P^TC_rp_r
$$

ALS 推荐分为：

$$
ALSScore(u,r)=P_u^TQ_r+b_r
$$

其中 $b_r$ 是由正反馈次数归一化得到的物品流行偏置，用于缓解稀疏数据上的不稳定。

### 8.4 Logistic Fusion

`fusion_lr` 用逻辑回归学习证据分到正反馈概率的映射。输入特征为：

$$
x(u,r)=[PreferenceScore,HealthGoalScore,DiseaseScore,ContentScore,FeedbackScore,SemanticScore,QualityScore]
$$

训练标签来自 cutoff 前的 synthetic 反馈：`click/save/cook` 和权重大于 1.0 的事件为正样本，`skip/dislike`、负权重事件和随机未交互采样为负样本。

$$
FusionScore(u,r)=\sigma(\theta^Tx(u,r)+b)
$$

训练流程使用 `StandardScaler + LogisticRegression(class_weight="balanced")`。它用于离线评估 `fusion_lr`，不替代默认 CLI 推荐路径。

### 8.5 Content Feedback Baseline

`content_feedback` 是轻量混合 baseline。先计算内容分：

$$
ContentBase(u,r)=0.45PreferenceScore(u,r)+0.35ContentScore(u,r)+0.20QualityScore(r)
$$

再加入反馈分：

$$
ContentFeedbackScore(u,r)=0.80ContentBase(u,r)+0.20FeedbackScore(u,r)
$$

### 8.6 DACH Grid Weight Search

`dach_grid` 不是新的深度模型，而是在 DACH 可解释证据项上做权重搜索。它先为验证用户缓存候选菜谱和 evidence 特征矩阵，再枚举多组权重，在验证集上选择 Top-K 指标最好的权重组合。

参与搜索的证据项为：

$$
f(u,r)=[PreferenceScore,HealthGoalScore,ContentScore,FeedbackScore,SemanticScore,QualityScore,DiversityBoost]
$$

给定一组权重 $w$，菜谱分数为：

$$
Score_{grid}(u,r)=\sum_{k}w_k^* f_k(u,r),\quad \sum_k w_k^*=1
$$

权重候选集合 $\mathcal{W}$ 由 `grid_step` 离散枚举得到，并限制单个分量不超过 `max_component_weight`；默认 DACH 权重也会额外加入候选集合。每组权重都按验证用户计算 Precision@K、Recall@K 和 NDCG@K，最终选择规则为：

$$
w^*=\arg\max_{w\in\mathcal{W}}(NDCG_K(w), Recall_K(w), Precision_K(w))
$$

也就是说，先比较 NDCG@K；如果并列，再比较 Recall@K；仍并列时比较 Precision@K。推荐时仍然执行 DACH 硬过滤，并默认排除用户已交互菜谱。候选池大小默认为 `max(200, 25K)`。`dach_grid` 只是把固定权重换成搜索权重，不会改变证据项定义。

## 9. 推荐执行步骤

### 9.1 默认菜谱推荐

1. 读取用户画像和用户语义向量。
2. 如果加载了 BPR 模型，对所有菜谱批量计算 BPR 分数。
3. 遍历菜谱并执行硬过滤。
4. 计算每个候选菜谱的 evidence。
5. 按 DACH 固定权重得到初始综合分。
6. 取前 `max(200, 25K)` 个候选进入多样性重排池。
7. 贪心选出 Top-K，每次选择后重新计算剩余候选的 `DiversityBoost`。
8. 返回推荐结果和证据解释。

### 9.2 食材推荐

1. 读取用户画像和用户语义向量。
2. 遍历食材，排除用户避免食材和可选疾病禁忌食材。
3. 计算食材口味、健康目标、疾病、内容、反馈、语义和质量证据。
4. 按食材排序公式得到综合分。
5. 返回 Top-K 食材及解释。

## 10. 评估方法

离线评估采用时间切分：

$$
Train=\{(u,r,e)\mid event\_time<cutoff\}
$$

$$
Test^+=\{(u,r,e)\mid event\_time\ge cutoff,\ e\in\{click,save,cook\}\}
$$

默认评估 ranker：

| ranker | 含义 |
| --- | --- |
| popularity | 训练期流行度 |
| content | 口味 + 内容 + 质量 baseline |
| content_feedback | 内容 baseline + 反馈分 |
| itemknn | 物品协同过滤 |
| als_only | 隐式反馈 ALS |
| bpr_only | BPR 个性化排序 |
| fusion_lr | 逻辑回归学习证据融合 |
| dach_grid | 验证集 NDCG@K 选择 DACH evidence 权重 |
| dach_no_health | DACH 去掉健康目标 |
| dach_no_llm | DACH 去掉语义分 |
| dach_no_feedback | DACH 去掉反馈分 |
| dach_no_diversity | DACH 去掉多样性 |
| dach_full | 完整 DACH |

评估指标包括 Precision@K、Recall@K、NDCG@K、HitRate@K、Coverage、Diversity 和 SafetyViolationRate。

由于反馈来自 synthetic 表，结果只能作为模拟实验或工程验收，不能作为真实用户有效性结论。

## 11. 当前实现状态

已经实现：默认 DACH 可解释推荐、食谱和食材双任务推荐、疾病约束 opt-in 扩展、HashEmbeddingProvider 离线语义向量、BPR 训练/偏置项/批量打分/Top-K、ItemKNN、隐式反馈 ALS、Logistic Regression 证据融合、DACH 权重网格搜索、content/content_feedback/popularity baseline、DACH 消融实验、BPR 诊断和一键实验输出、常见 Top-K 指标和安全违规率。

仍未实现或不能声称完成：真实大模型 embedding、大模型生成解释、大模型 reranker、LightGCN/GAT/DeepFM/Wide&Deep 等深度或图模型、真实用户实验、基于真实疾病诊断的个性化推荐、临床营养或疾病治疗建议、多随机种子显著性检验和置信区间报告、生产级 API 服务。

## 12. 总结

优化后的 DACH-LLMRec 保留可解释健康约束排序作为默认路径，同时新增 BPR、ItemKNN、ALS、Logistic Fusion 和 DACH 权重网格搜索作为协同过滤、学习型融合与权重调参对比。这样既能保持推荐结果的证据可追溯性，也能用更完整的离线实验验证不同个性化模型在 synthetic 反馈上的表现。
