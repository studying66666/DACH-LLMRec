# DACH-LLMRec 投稿升级文献矩阵

检索日期：2026-07-31。该矩阵为筛查级材料，不等同于系统综述。

| 文献 | 类型 | 主要贡献 | 与 DACH-LLMRec 的关系 | 对本项目留下的空间 |
|---|---|---|---|---|
| Zhang et al., CoLLM, IEEE TKDE 2025, arXiv:2310.19488, DOI:10.1109/TKDE.2025.3540912 | LLM4Rec 方法 | 将传统推荐模型中的协同 embedding 映射到 LLM token embedding 空间。 | 说明协同信号不应被纯文本 LLM prompting 取代。 | 非食品领域；未聚焦健康约束、长期膳食轨迹或忠实营养解释。 |
| Kim et al., Lost in Sequence / LLM-SRec, KDD 2025, arXiv:2502.13909 | 序列 LLM4Rec 诊断与方法 | 指出已有 LLM4Rec 方法常不能理解用户交互序列，并通过轻量 MLP 将 CF-SRec 表征蒸馏进 LLM。 | 直接支撑 DACH-LLMRec 增加序列感知知识蒸馏。 | 面向通用序列推荐，不是健康感知食物推荐；缺少营养约束轨迹。 |
| Wei et al., LLMRec, WSDM 2024, arXiv:2311.00423, DOI:10.1145/3616855.3635853 | LLM 图增强 | 使用 LLM 增强用户-物品交互边、物品属性和用户画像。 | 可作为 LLM 数据增强和图增强推荐的先例。 | 评估主要在娱乐领域；没有硬营养约束或医学/健康证据边界。 |
| Zhang et al., DualAgent-Rec, WWW Companion 2026, arXiv:2601.19121, DOI:10.1145/3774905.3795728 | 约束合规多智能体优化 | 使用 LLM 协调器在硬约束下分配 exploitation 与 exploration 智能体。 | 支撑“LLM 作为优化/调度器，而非直接打分器”的定位。 | 面向电商约束，不是营养约束；无序列膳食状态或营养平衡。 |
| Lechiakh et al., Towards long-term depolarized interactive recommendations, Information Processing & Management 2024, DOI:10.1016/j.ipm.2024.103833 | 约束强化学习推荐 | 用 DQN 变体和类别级约束建模长期交互推荐。 | 是长期约束推荐的近邻模板。 | 优化目标是去极化，不是膳食平衡；未结合 LLM，也非食品领域。 |
| Sani et al., FIRE, arXiv:2508.05225 | 忠实解释 | 用特征归因和 LLM 生成，将推荐解释锚定在模型决策逻辑上。 | 支撑 DACH 的证据约束解释与解释忠实性指标。 | 通用评论领域解释；未处理营养约束或膳食证据。 |
| Zhang et al., MOPI-HFRS, KDD 2025, arXiv:2412.08847, DOI:10.1145/3690624.3709382 | 健康感知食物推荐 | 构建健康感知食物推荐 benchmark、图结构学习、Pareto 优化和 LLM 增强解释。 | 是“健康感知食物推荐 + LLM 解释”主张的最强直接竞争工作。 | 更偏静态 Top-K 健康感知食物推荐；本项目可在序列膳食轨迹、硬约束和解释忠实性上区分。 |
| Khamesian et al., NutriGen, EMBC 2025 / arXiv:2502.20601, DOI:10.1109/EMBC58623.2025.11253879 | LLM 膳食计划生成 | 使用 LLM prompting 和营养数据库生成符合偏好与约束的个性化膳食计划。 | 可作为膳食计划生成的实用基线。 | 偏生成式计划；协同推荐、Top-K 排序和模型证据忠实解释较弱。 |
| Papastratis et al., Scientific Reports 2024, DOI:10.1038/s41598-024-65438-x | 深度生成式营养规划 | 使用深度生成模型和 ChatGPT，并结合营养指南损失进行膳食规划。 | 支撑营养指南损失设计。 | 是膳食计划生成而非推荐系统序列策略；协同信号有限。 |
| Yang et al., ChatDiet, Smart Health 2024, DOI:10.1016/j.smhl.2024.100465 | 对话式营养推荐 | 构建 LLM 增强的营养导向食物推荐聊天机器人，结合个人模型与群体模型。 | 支撑实用对话界面和个性化方向。 | 更偏案例研究；解释质量和约束合规仍需更强模型证据约束。 |
| Marin et al., Recipe1M+, IEEE TPAMI 2019, DOI:10.1109/TPAMI.2019.2927476 | 数据集 / 多模态食物表征 | 提供大规模菜谱-图像数据集和跨模态 embedding。 | 可作为外部菜谱语义和多模态特征数据源。 | 不包含纵向用户健康结果；营养覆盖有限。 |
| Food.com Recipes and Interactions, Kaggle | 数据集 | 提供 Food.com 菜谱与长期用户交互/评论。 | 可作为序列用户-菜谱交互的外部实验数据。 | 健康标签和营养约束需要额外增强与验证。 |

