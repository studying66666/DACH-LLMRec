# DACH-LLMRec 投稿升级选题备忘录

## 1. 执行摘要

| 字段 | 内容 |
|---|---|
| 研究领域 | LLM 增强的健康感知食物推荐 |
| 编制日期 | 2026-07-31 |
| 结论等级 | 筛查级：整理证据，但最终是否立项由研究者和导师决定 |
| 检索可信度 | 中等：已检查关键直接竞争工作，但不是系统综述 |

| 候选方向 1 | 长期约束忠实营养推荐 |
|---|---|
| 结论 | 值得推进，但必须满足开放条件 |
| 原因 | 静态“健康感知食物推荐 + LLM 解释”已被 MOPI-HFRS 占据相当空间。更可防守的开口是：序列感知、长期营养约束和基于证据的忠实解释。 |

**关键不确定性。** 项目必须超越 synthetic 用户和 hash embedding；否则它更像工程原型，而不是一区论文级研究贡献。

## 2. 候选方向定义

**长期约束忠实营养推荐。** 将 DACH-LLMRec 升级为序列感知、长期健康约束的食谱推荐框架：把协同/序列用户表征蒸馏为 LLM 可读的语义状态，在多餐周期内联合优化偏好、健康、多样性与安全，并且只基于已验证模型证据生成解释。*为什么可能是研究空白：* 当前方法仍有不足，因为近邻工作通常只解决其中一部分：协同 LLMRec、图增强、序列推荐、约束优化、食物推荐或忠实解释。

## 3. 决策评分表

| 关口 | 分数 | 理由 |
|---|---:|---|
| 关口 1 - 空白是否仍开放 | 3/5 | 宽泛方向已被 MOPI-HFRS、NutriGen、ChatDiet、CoLLM、LLMRec、LLM-SRec、DualAgent-Rec 和 FIRE 部分占据。本轮筛查显示，“长期序列 + 硬营养约束 + 模型证据忠实解释”这一精确交叉点仍只是部分占据。 |
| 关口 2 - 是否构成真实贡献 | 4/5 | 如果能实现为带 benchmark、长期健康指标和可复现实验的框架，它更接近解决问题型贡献，而不是简单拼接。 |
| 关口 3 - 是否可行 | 3/5 | 可行但需要投入：Food.com 和 Recipe1M+ 可提供物品与交互信号，本地 DACH 数据库可提供结构化健康/食材知识，但营养归一化和用户健康标签需要仔细工程处理。 |
| 结论 | 条件性推进 | 只有在 pilot 证明数据对齐、序列评估和解释忠实性指标可复现后，才建议正式投入。 |

## 4. 具体论文方案

**建议题目。** DACH-SRec: Longitudinal Constraint-Faithful LLM-Enhanced Recommendation for Personalized Nutrition。

**核心研究问题。** 膳食推荐系统如何在多餐序列上同时优化用户偏好、健康目标达成、多样性和安全性，并生成忠实于排序证据的自然语言解释？

**方法设计。** 保留当前 DACH 的硬过滤和 evidence scoring 作为安全层。新增 SASRec/GRU4Rec 等序列编码器，在 Food.com 用户-菜谱历史上训练。参考 LLM-SRec 和 CoLLM，用轻量 MLP adapter 将用户序列表征蒸馏到 LLM/embedding 语义空间。增加约束重排或轻量 RL 层，在日/周级 slate 中同时满足热量、宏量营养素、避免食材、健康目标、多样性和重复率约束。LLM 只承担两个角色：增强菜谱/用户画像语义表示，以及把结构化证据润色为解释；不让 LLM 作为不可控的最终排序器。

**主要创新主张。**

1. 提出长期健康约束食谱推荐形式化定义：状态不再是单个用户-物品对，而是近期膳食轨迹。
2. 提出协同/序列表征到 LLM 语义空间的轻量 adapter，在不微调整个 LLM 的情况下注入用户历史。
3. 设计 meal slate 级约束合规优化层，报告硬约束零违规或近零违规。
4. 提出忠实解释模块，解释必须引用排序器真实使用的 evidence 字段和 attribution 分数。
5. 构建可复现 benchmark，将 Food.com/Recipe1M+ 交互与 DACH 营养、食材、健康目标知识连接。

## 5. 一区论文所需实验

| 实验 | 必要细节 |
|---|---|
| 数据集构建 | 尽可能对齐本地 DACH 菜谱/食材与 Food.com/Recipe1M+；报告用户数、菜谱数、交互数、稀疏度、营养覆盖率、约束覆盖率和训练/验证/测试划分。 |
| Baseline | Popularity、ItemKNN、BPR、LightGCN、SASRec/GRU4Rec、LLMRec 风格增强、CoLLM 风格协同 adapter、可复现时加入 MOPI-HFRS 风格多目标图 baseline、NutriGen/LLM prompt planner 作为膳食计划对比。 |
| 准确性指标 | Recall@K、NDCG@K、HitRate@K、MRR、冷/暖启动分组。 |
| 健康指标 | HealthGoalHit@K、营养目标偏差、避免食材违规率、周重复率、长期健康平衡分。 |
| 多目标指标 | Pareto hypervolume、约束满足率、覆盖率、多样性、偏好与健康之间的 trade-off 曲线。 |
| 解释指标 | 证据精确率、特征归因一致性、幻觉声明率、情感/健康一致性；如条件允许，加入用户或营养师评价。 |
| 鲁棒性 | 多随机种子、显著性检验、消融实验、API/模型版本记录、prompt 与 LLM 输出归档。 |

## 6. 最小实现路线

**阶段 1：数据与 benchmark，2-3 周。** 新增 Food.com 外部数据 loader。归一化食材名称和营养字段。生成 `user -> recipe sequence` 表和可复现时间切分。通过营养规则和 DACH HCI 映射补充健康目标标签。

**阶段 2：序列协同模型，2-4 周。** 实现 SASRec 或 GRU4Rec baseline。新增序列感知 DACH 分数 `SeqScore(u,r)`。与当前 BPR/ALS/ItemKNN 对比。

**阶段 3：LLM/语义 adapter，2-4 周。** 用真实本地或 API embedding provider 替换 hash embedding。训练轻量 adapter，对齐序列用户状态、协同物品状态和菜谱文本 embedding。

**阶段 4：约束优化器，2-4 周。** 先实现 slate 级 constrained reranking；如果重排无法处理长期累计约束，再考虑 RL。精确记录硬约束违规。

**阶段 5：忠实解释，2 周。** 只从 evidence JSON 生成解释：最高分因素、约束检查结果、attribution 值。增加自动幻觉检查，验证解释中的每个声明都能在 evidence 字段中找到依据。

**阶段 6：论文级评估，3-5 周。** 跑完整 baseline、消融、显著性检验、案例分析和效率分析。冻结 prompt、模型版本、随机种子和数据划分。

## 7. 停止条件

如果以下任一条件失败，不建议把论文定位为一区就绪：

1. 外部数据无法提供可信的用户-菜谱序列历史。
2. 方法不能在 NDCG@K 上超过 SASRec/LightGCN 类 baseline，或者只能靠牺牲健康约束换取精度。
3. 硬约束满足率没有明显优于软惩罚 baseline。
4. 解释中出现 evidence 字段不存在的健康或营养声明。
5. 研究仍然只依赖 synthetic 用户。

## 8. 目标期刊与会议

如果方法和实验足够强，优先考虑：IEEE TKDE、Information Fusion、Expert Systems with Applications、Knowledge-Based Systems、ACM TORS、Information Processing & Management。若先做较短应用版本，可考虑 KDD Applied Data Science、WSDM、RecSys 或 WWW Companion；但一区期刊投稿需要更强实验和更清晰 benchmark 叙事。

