# DACH-LLMRec 算法说明

本文档说明当前项目实际实现的推荐算法。重点是把算法逻辑、数学公式、执行步骤和使用的方法讲清楚，而不是列代码接口。

当前 DACH-LLMRec 是一个健康目标约束的混合推荐原型。它把结构化规则、用户画像、食材/菜谱内容、隐式反馈、语义向量和多样性重排组合起来，完成食谱和食材 Top-K 推荐。

需要先明确三条边界：

1. 当前用户画像和反馈来自 synthetic 表，只能用于模拟实验。
2. 当前默认语义向量是确定性哈希向量，不是真实大模型 embedding。
3. 疾病相关知识表默认不代表用户诊断；只有显式传入疾病 ID 时，疾病硬过滤和疾病分数才启用。

## 1. 总体方法

DACH-LLMRec 不是单一模型，而是一个“先过滤、再打分、再重排、最后解释”的多证据融合排序方法。

整体逻辑如下：

1. 从 SQLite 膳食数据库读取菜谱、食材、菜谱-食材关系、口味知识、健康目标知识、营养可信度、用户画像和用户反馈。
2. 构建用户画像，包括口味偏好、健康目标、偏好食材、避免食材、避免菜谱和历史行为。
3. 构建候选项特征，包括菜谱口味分布、菜谱健康目标信号、食材健康目标信号、内容质量和语义文本向量。
4. 在排序前执行硬过滤，先移除不可推荐项、用户避免项和可选疾病禁忌项。
5. 对剩余候选计算多项证据分数。
6. 将证据分数加权融合为综合分数。
7. 对菜谱 Top-K 做贪心多样性重排，降低同菜系、同做法、同主食材的重复。
8. 根据证据阈值生成模板化解释。

当前实际用到的方法包括：

- 基于规则的安全硬过滤；
- 基于口味向量的内容相似度；
- 基于 HCI 健康目标知识的规则匹配；
- 基于用户偏好食材的内容匹配；
- 基于隐式反馈事件权重的行为分数；
- 可选 BPR 隐式反馈矩阵分解；
- 基于文本向量余弦相似度的语义匹配；
- 基于内容完整度和营养可信度的质量分；
- 基于菜系、做法和主食材相似度的多样性重排。

## 2. 符号定义

设用户为 \(u\)，菜谱为 \(r\)，食材为 \(i\)，健康目标为 \(c\)，疾病或健康风险因素为 \(d\)。

主要集合和向量：

- \(T_u\)：用户口味偏好向量。
- \(T_r\)：菜谱口味分布向量。
- \(H_u\)：用户健康目标权重集合。
- \(I_r\)：菜谱 \(r\) 包含的食材集合。
- \(F_u\)：用户偏好食材集合及权重。
- \(A_u^r\)：用户避免的菜谱集合。
- \(A_u^i\)：用户避免的食材集合。
- \(E_u\)：用户文本画像向量。
- \(E_r\)：菜谱文本向量。
- \(E_i\)：食材文本向量。

所有证据分数最终都归一化到 \([0,1]\)。分数越高，表示该证据越支持推荐。

## 3. 用户画像构建

### 3.1 口味偏好

用户口味偏好来自 synthetic 用户口味表。原始偏好取值为 \(-2,-1,0,1,2\)，表示从强烈不喜欢到强烈喜欢。

归一化后：

\[
T_u(t)=\frac{p_u(t)}{2}
\]

其中 \(p_u(t)\) 是用户对口味 \(t\) 的原始偏好值。归一化后 \(T_u(t)\in[-1,1]\)。

### 3.2 健康目标权重

用户健康目标来自 synthetic 用户健康目标表。系统假设优先级数字越小，健康目标越重要。

\[
H_u(c)=\frac{1}{priority_u(c)}
\]

例如，优先级为 1 的目标权重为 1.0，优先级为 2 的目标权重为 0.5。

健康目标表有父子层级。为了让父目标和子目标都能参与匹配，系统会扩展用户健康目标：

\[
H_u(c_{child})=\max(H_u(c_{child}),0.8H_u(c_{parent}))
\]

\[
H_u(c_{parent})=\max(H_u(c_{parent}),0.5H_u(c_{child}))
\]

含义是：父目标向子目标传播时保留 80% 权重，子目标向父目标传播时保留 50% 权重。重复传播后保留最大权重。

### 3.3 历史反馈

用户反馈事件使用固定行为权重：

| 事件 | 权重 |
| --- | ---: |
| cook | 5.0 |
| save | 4.0 |
| click | 2.0 |
| impression | 0.5 |
| skip | -1.0 |
| dislike | -4.0 |

对用户 \(u\) 和菜谱 \(r\)，事件聚合值为：

\[
RawFeedback(u,r)=\sum_{e\in Events(u,r)}w_e
\]

其中 \(w_e\) 是事件权重。

## 4. 候选项特征构建

### 4.1 菜谱食材权重

菜谱和食材之间的关系来自菜谱-食材表。系统区分主料和辅料：

\[
W(r,i)=
\begin{cases}
1.0, & i\text{ 是主料}\\
0.5, & i\text{ 是辅料}
\end{cases}
\]

如果同一食材在同一菜谱中重复出现，保留最大权重。

### 4.2 菜谱口味向量

菜谱口味不是直接由用户反馈得到，而是通过“菜谱 -> 食材 -> 口味”的知识链路统计得到。

对菜谱 \(r\) 和口味 \(t\)：

\[
T_r(t)=\frac{\sum_{i\in I_r}W(r,i)\cdot \mathbb{1}(i\text{ 具有口味 }t)}
{\sum_{i\in I_r}W(r,i)}
\]

也就是说，主料对菜谱口味分布影响更大，辅料仍参与但权重较低。

### 4.3 健康目标信号

菜谱健康目标信号有两种来源。

直接信号来自健康目标推荐菜谱关系：

\[
DirectHealth(r,c)=\frac{Intensity(r,c)}{5}
\]

间接信号来自健康目标推荐食材关系：

\[
IngredientHealth(i,c)=\frac{Intensity(i,c)}{5}
\]

其中 \(Intensity\) 是数据库中的推荐强度，当前按 5 做归一化，并截断到 \([0,1]\)。

## 5. 硬过滤

硬过滤在所有加权打分之前执行。被硬过滤排除的候选不会进入排序，因此不能被高兴趣分数抵消。

菜谱 \(r\) 必须同时满足：

\[
recommendable(r)=1
\]

\[
r\notin A_u^r
\]

\[
I_r\cap A_u^i=\varnothing
\]

如果显式启用疾病因素 \(d\)，还必须满足：

\[
r\notin AvoidRecipe(d)
\]

\[
I_r\cap AvoidIngredient(d)=\varnothing
\]

食材推荐也会排除用户避免食材；启用疾病因素时，还会排除疾病禁忌食材。

## 6. 多证据分数

### 6.1 口味匹配分

菜谱口味匹配分使用用户口味向量和菜谱口味向量的余弦相似度：

\[
PreferenceScore(u,r)=\frac{\cos(T_u,T_r)+1}{2}
\]

如果用户没有口味画像，或菜谱没有可用口味向量，则使用中性分 0.5。

食材口味分取该食材关联口味在用户口味向量中的平均偏好，再映射到 \([0,1]\)：

\[
PreferenceScore(u,i)=\frac{1}{2}+\left(\frac{1}{2|T_i|}\sum_{t\in T_i}T_u(t)\right)
\]

无口味证据时同样使用 0.5。

### 6.2 健康目标分

菜谱健康目标分由直接匹配和食材间接匹配组成。

直接匹配：

\[
Direct(u,r)=\max_{c\in H_u} H_u(c)\cdot DirectHealth(r,c)
\]

间接匹配：

\[
Indirect(u,r)=\frac{1}{|I_r|}\sum_{i\in I_r}\max_{c\in H_u} H_u(c)\cdot IngredientHealth(i,c)
\]

最终菜谱健康目标分：

\[
HealthGoalScore(u,r)=clip_{[0,1]}\left(0.6Direct(u,r)+0.4Indirect(u,r)\right)
\]

食材健康目标分：

\[
HealthGoalScore(u,i)=clip_{[0,1]}\left(\max_{c\in H_u}H_u(c)\cdot IngredientHealth(i,c)\right)
\]

没有用户健康目标时使用中性分 0.5。

### 6.3 内容偏好分

菜谱内容偏好分衡量菜谱食材是否命中用户偏好食材。

\[
ContentScore(u,r)=\frac{\sum_{i\in I_r}W(r,i)\cdot F_u(i)}{\sum_{i\in I_r}W(r,i)}
\]

其中 \(F_u(i)\) 是用户对食材 \(i\) 的偏好权重。没有偏好食材证据时使用 0.5。

食材内容偏好分直接使用用户对该食材的偏好权重；如果没有记录，则使用 0.5：

\[
ContentScore(u,i)=F_u(i)
\]

### 6.4 反馈分

先将用户对菜谱的历史行为聚合为事件分：

\[
EventScore(u,r)=\sigma\left(\frac{RawFeedback(u,r)}{5}\right)
\]

其中 \(\sigma(x)=\frac{1}{1+e^{-x}}\)。没有历史行为时使用 0.5。

如果没有加载 BPR 模型：

\[
FeedbackScore(u,r)=EventScore(u,r)
\]

如果加载了 BPR 模型：

\[
BPRScore(u,r)=\sigma(P_u^\top Q_r)
\]

\[
FeedbackScore(u,r)=clip_{[0,1]}\left(0.5EventScore(u,r)+0.5BPRScore(u,r)\right)
\]

其中 \(P_u\) 是 BPR 学到的用户隐向量，\(Q_r\) 是菜谱隐向量。

食材反馈分不是单独训练得到，而是把包含该食材的菜谱反馈分取平均：

\[
FeedbackScore(u,i)=\frac{1}{|\mathcal{R}(i)|}\sum_{r\in\mathcal{R}(i)}FeedbackScore(u,r)
\]

其中 \(\mathcal{R}(i)\) 表示包含食材 \(i\) 且用户有反馈记录的菜谱集合。集合为空时使用 0.5。

### 6.5 语义匹配分

系统把用户画像、菜谱信息和食材信息拼接成文本，再通过 embedding provider 得到向量。

用户文本包含年龄、性别、活动水平、饮食目标、口味、健康目标、运动和偏好食材。

菜谱文本包含菜谱名称、描述、菜系、做法、口味标签、食材和营养可信度。

食材文本包含食材名称、类别、营养状态和来源状态。

语义匹配分为：

\[
SemanticScore(u,x)=\frac{\cos(E_u,E_x)+1}{2}
\]

其中 \(x\) 可以是菜谱 \(r\)，也可以是食材 \(i\)。

当前代码字段名为 `llm_alignment_score`，但默认向量来自本地 HashEmbeddingProvider，不是真实 LLM embedding。只有替换为真实 embedding provider 后，才可以把这一项表述为真实大模型语义分。

### 6.6 质量分

菜谱质量分由内容状态和营养可信度相乘得到：

\[
QualityScore(r)=ContentStatusScore(r)\cdot NutritionTierScore(r)
\]

内容状态分：

| 内容状态 | 分数 |
| --- | ---: |
| complete | 1.0 |
| partial | 0.6 |
| sparse | 0.3 |
| 其他或缺失 | 0.5 |

营养可信度分：

| 营养可信度 | 分数 |
| --- | ---: |
| standard | 1.0 |
| sensitivity_only | 0.6 |
| exclude_from_nutrition_model | 0.2 |
| 其他或缺失 | 0.5 |

食材质量分更简单：营养状态为 observed 时取 1.0，否则取 0.6。

### 6.7 疾病扩展分

疾病扩展分默认关闭。只有调用方显式传入疾病 ID 并启用疾病约束时，该分数才进入综合排序。

菜谱疾病推荐分也由直接信号和食材间接信号组成：

\[
DirectDisease(r,d)=RecommendRecipe(d,r)
\]

\[
IndirectDisease(r,d)=\frac{1}{|I_r|}\sum_{i\in I_r}RecommendIngredient(d,i)
\]

多疾病 ID 时，直接部分对疾病 ID 取最大值；间接部分对每个食材先取最大疾病推荐强度，再对菜谱食材平均：

\[
DirectDisease(r)=\max_d RecommendRecipe(d,r)
\]

\[
IndirectDisease(r)=\frac{1}{|I_r|}\sum_{i\in I_r}\max_d RecommendIngredient(d,i)
\]

\[
DiseaseScore(r)=clip_{[0,1]}\left(0.6DirectDisease(r)+0.4IndirectDisease(r)\right)
\]

食材疾病推荐分为：

\[
DiseaseScore(i)=\max_d RecommendIngredient(d,i)
\]

注意：疾病禁忌项不是低分，而是在硬过滤阶段直接排除；疾病推荐项才进入 DiseaseScore。

### 6.8 多样性增益

多样性只用于菜谱 Top-K 重排。系统计算候选菜谱与已选菜谱的最大相似度，再把相似度转成多样性增益。

两个菜谱的相似度为：

\[
sim(r,s)=0.4C(r,s)+0.3J(M_r,M_s)+0.3J(G_r,G_s)
\]

其中：

- \(C(r,s)=1\) 表示两个菜谱菜系相同，否则为 0；
- \(M_r\) 是菜谱 \(r\) 的做法集合；
- \(G_r\) 是菜谱 \(r\) 的主食材集合；
- \(J(\cdot,\cdot)\) 是 Jaccard 相似度。

候选菜谱的多样性增益为：

\[
DiversityBoost(r)=1-\max_{s\in Selected}sim(r,s)
\]

如果还没有已选菜谱，则 \(DiversityBoost(r)=1.0\)。

## 7. 综合排序公式

### 7.1 默认菜谱排序

默认不启用疾病因素时，菜谱综合分为：

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

### 7.2 启用疾病因素后的菜谱排序

显式启用疾病因素后，综合分为：

\[
\begin{aligned}
Score(u,r)=&
0.18PreferenceScore(u,r)
+0.18HealthGoalScore(u,r)\\
&+0.16DiseaseScore(r)
+0.14ContentScore(u,r)\\
&+0.12FeedbackScore(u,r)
+0.10SemanticScore(u,r)\\
&+0.08QualityScore(r)
+0.04DiversityBoost(r)
\end{aligned}
\]

这不是默认路径。当前项目不能从数据库自动推断用户疾病，只能使用调用方显式传入的疾病 ID。

### 7.3 默认食材排序

默认不启用疾病因素时，食材综合分为：

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

### 7.4 启用疾病因素后的食材排序

\[
\begin{aligned}
IngredientScore(u,i)=&
0.18PreferenceScore(u,i)
+0.18HealthGoalScore(u,i)\\
&+0.18DiseaseScore(i)
+0.18ContentScore(u,i)\\
&+0.08FeedbackScore(u,i)
+0.10SemanticScore(u,i)\\
&+0.10QualityScore(i)
\end{aligned}
\]

## 8. 推荐执行步骤

### 8.1 菜谱推荐步骤

1. 读取用户 \(u\) 的 synthetic 画像、口味、健康目标、偏好食材、避免食材、避免菜谱和历史反馈。
2. 扩展用户健康目标层级，得到 \(H_u\)。
3. 遍历所有菜谱，先执行硬过滤。
4. 对通过过滤的每个菜谱计算 PreferenceScore、HealthGoalScore、ContentScore、FeedbackScore、SemanticScore、QualityScore 和初始 DiversityBoost。
5. 如果启用疾病因素，再计算 DiseaseScore，并使用疾病扩展权重。
6. 先按当前综合分得到候选池。
7. 从候选池中贪心选择 Top-K。每选出一个菜谱，就重新计算剩余候选对已选集合的 DiversityBoost，再更新综合分。
8. 返回 Top-K、综合分、各证据分、命中因素和模板化解释。

### 8.2 食材推荐步骤

1. 读取用户画像和用户文本向量。
2. 遍历所有食材，排除用户避免食材；启用疾病因素时，再排除疾病禁忌食材。
3. 对每个候选食材计算口味分、健康目标分、内容偏好分、反馈分、语义分和质量分。
4. 如果启用疾病因素，再计算食材疾病推荐分。
5. 按综合分排序，返回 Top-K、证据和解释。

## 9. BPR 隐式反馈学习

BPR 用于学习用户和菜谱的潜在偏好。它不替代主排序公式，而是作为 FeedbackScore 的一部分参与融合。

正反馈事件：click、save、cook。

负反馈事件：skip、dislike。

训练样本是三元组 \((u,r^+,r^-)\)，表示用户 \(u\) 对正样本菜谱 \(r^+\) 的偏好应该高于负样本菜谱 \(r^-\)。

BPR 打分：

\[
\hat{y}_{u,r}=P_u^\top Q_r
\]

BPR 损失：

\[
L_{BPR}=-\frac{1}{|\mathcal{D}|}\sum_{(u,r^+,r^-)\in\mathcal{D}}\log\sigma(\hat{y}_{u,r^+}-\hat{y}_{u,r^-})
\]

其中 \(\mathcal{D}\) 是训练三元组集合。

当前训练使用 AdamW，包含优化器权重衰减。模型保存用户索引、菜谱索引、用户隐向量、菜谱隐向量、训练配置和每轮损失。推荐阶段加载模型后，将点积分数经 sigmoid 映射为 \(BPRScore(u,r)\)，再融合到 FeedbackScore。

当前没有实现 LightGCN、GAT、DeepFM、Wide&Deep，也没有实现语义对齐损失、健康边界损失或安全惩罚损失。这些只能写成后续增强方向。

## 10. 评估方法

离线评估采用时间切分。训练反馈满足：

\[
event\_time < cutoff
\]

测试正样本满足：

\[
event\_time \ge cutoff,\quad event\_type\in\{click,save,cook\}
\]

默认比较方法包括 popularity、content、bpr_only、dach_no_health、dach_no_llm、dach_no_feedback、dach_no_diversity 和 dach_full。

评估指标包括 Precision@K、Recall@K、NDCG@K、HitRate@K、Coverage、Diversity 和 SafetyViolationRate。

因为反馈来自 synthetic 表，评估结果只能称为模拟实验或 demo 验收，不能称为真实用户实验。

## 11. 当前实现状态

已经实现：SQLite 数据读取和 demo 数据生成、用户画像构建、HCI 健康目标层级扩展、菜谱和食材特征构建、硬过滤、菜谱与食材多证据加权公式、HashEmbeddingProvider 离线语义向量、BPR 隐式反馈训练与加载、baseline/消融评估、常见 Top-K 指标，以及基于证据字段的模板化解释。

未实现，不能声称完成：真实大模型 embedding、大模型解释生成、大模型 reranker、LightGCN/PinSage/GAT/异构图神经网络、Two-Tower/DeepFM/Wide&Deep、真实用户实验、基于真实疾病诊断的个性化推荐、临床营养或疾病治疗建议、多随机种子显著性检验和生产级 API 服务。

## 12. 一句话总结

当前 DACH-LLMRec 的核心不是让大模型直接推荐菜，而是把健康推荐场景中的安全过滤、结构化健康目标、食材内容、隐式反馈、语义相似度、质量控制和多样性放进一个可解释的排序框架中。它适合作为可复现研究原型；如果要写成正式论文，还需要真实 embedding、完整数据实验、多种强 baseline 和严格统计分析。
