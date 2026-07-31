# DACH-LLMRec 投稿升级选题备忘录

## 1. Executive Decision Summary

| Field | Value |
|---|---|
| Area | LLM-enhanced health-aware food recommendation |
| Compiled | 2026-07-31 |
| Verdict grade | Screening-grade: evidence assembled; final go/no-go belongs to researcher/advisor |
| Search confidence | Medium: key direct competitors were checked, but this is not a systematic review |

| Candidate 1 | Longitudinal constraint-faithful nutrition recommendation |
|---|---|
| Verdict | Worth pursuing, only if its open conditions hold |
| Reason | Static "health-aware food recommendation + LLM explanation" is already occupied by MOPI-HFRS. The defensible opening is narrower: sequence-aware, long-horizon nutrition constraints plus faithful evidence-grounded explanations. |

**Key uncertainty.** The project must move beyond synthetic users and hash embeddings; otherwise it is an engineering prototype rather than a Q1-ready research contribution.

## 2. Candidate Definition

**Longitudinal constraint-faithful nutrition recommendation.** Upgrade DACH-LLMRec into a sequence-aware, long-term health-constrained recipe recommendation framework that distills collaborative/sequential user representations into LLM-readable semantic states, optimizes preference-health-diversity over a multi-meal horizon, and generates explanations only from verified model evidence. *Why it could be a gap:* Current methods fall short because nearby works usually solve only one part: collaborative LLMRec, graph augmentation, sequential recommendation, constrained optimization, food recommendation, or faithful explanation.

## 3. Decision Scorecard

| Gate | Score | Rationale |
|---|---:|---|
| Gate 1 - Gap still open | 3/5 | The broad idea is partially occupied by MOPI-HFRS, NutriGen, ChatDiet, CoLLM, LLMRec, LLM-SRec, DualAgent-Rec, and FIRE. The precise intersection of long-term sequence + hard nutrition constraints + faithful model-grounded explanations remains only partially occupied in this screening search. |
| Gate 2 - Real contribution | 4/5 | If implemented as a benchmarked framework with new long-horizon health metrics, this is a problem-solving contribution rather than a simple combination. |
| Gate 3 - Feasible | 3/5 | Feasible with effort: Food.com and Recipe1M+ can supply item and interaction signals, the local DACH database supplies structured health/ingredient knowledge, but nutrient normalization and user-health labeling require careful engineering. |
| Verdict | Conditional go | Pursue only after a pilot proves data alignment, sequence evaluation, and explanation-fidelity metrics can be made reproducible. |

## 4. Specific Paper Plan

**Recommended title.** DACH-SRec: Longitudinal Constraint-Faithful LLM-Enhanced Recommendation for Personalized Nutrition.

**Core research question.** How can a food recommender optimize user preference, health-goal adherence, diversity, and safety over a sequence of meals while producing natural-language explanations that are faithful to the ranking evidence?

**Method.** Keep the current DACH hard-filtering and evidence scoring as the safety layer. Add a sequence encoder, such as SASRec/GRU4Rec, trained on Food.com user-recipe histories. Distill its user sequence representation into the LLM/embedding space using lightweight MLP adapters, following the lesson of LLM-SRec and CoLLM. Add a constrained reranking or lightweight RL layer that chooses a daily or weekly slate under calorie, macronutrient, ingredient-avoidance, health-goal, diversity, and repetition constraints. Use the LLM only for two roles: semantic enrichment of recipe/user profiles and explanation verbalization from structured evidence, not as an unconstrained final ranker.

**Main novelty claims.**

1. A longitudinal health-constrained recipe recommendation formulation where the state is not a single user-item pair but a recent meal trajectory.
2. A collaborative-sequential-to-LLM adapter that injects user history into semantic recommendation without fine-tuning the full LLM.
3. A constraint-compliant optimization layer for meal slates, reporting zero or near-zero hard-constraint violations.
4. A faithful explanation module that cites the exact evidence fields and attribution scores used by the ranker.
5. A reproducible benchmark joining Food.com/Recipe1M+ interactions with DACH nutrition/ingredient/health-goal knowledge.

## 5. Experiments Needed for Q1

| Experiment | Required detail |
|---|---|
| Dataset construction | Align local DACH recipes/ingredients with Food.com/Recipe1M+ where possible; report users, recipes, interactions, sparsity, nutrient coverage, constraint coverage, and train/validation/test split. |
| Baselines | Popularity, ItemKNN, BPR, LightGCN, SASRec/GRU4Rec, LLMRec-style augmentation, CoLLM-style collaborative adapter, MOPI-HFRS-style multi-objective graph baseline if reproducible, NutriGen/LLM prompt planner for meal-plan comparison. |
| Accuracy metrics | Recall@K, NDCG@K, HitRate@K, MRR, cold/warm-start split. |
| Health metrics | HealthGoalHit@K, nutrient deviation from target, avoid-ingredient violation rate, weekly repetition rate, long-horizon health balance score. |
| Multi-objective metrics | Pareto hypervolume, constraint satisfaction rate, coverage, diversity, trade-off curves between preference and health. |
| Explanation metrics | Evidence precision, feature-attribution agreement, hallucinated-claim rate, sentiment/health consistency, human or dietitian preference study if available. |
| Robustness | Different random seeds, significance tests, ablations, API/model-version logging for embeddings and LLM explanations. |

## 6. Minimum Implementation Roadmap

**Phase 1: Data and benchmark, 2-3 weeks.** Build an external dataset loader for Food.com. Normalize ingredient names and nutrients. Create a `user -> recipe sequence` table and a reproducible split. Add health-goal labels through nutrient rules and DACH HCI mappings.

**Phase 2: Sequential collaborative model, 2-4 weeks.** Implement SASRec or GRU4Rec baseline. Add sequence-aware DACH score `SeqScore(u,r)`. Compare against current BPR/ALS/ItemKNN.

**Phase 3: LLM/semantic adapter, 2-4 weeks.** Replace hash embeddings with a real local or API embedding provider. Train a lightweight adapter to align sequential user states, collaborative item states, and recipe text embeddings.

**Phase 4: Constraint optimizer, 2-4 weeks.** Implement slate-level constrained reranking first; add RL only if reranking cannot model long-term accumulation. Track exact hard violations.

**Phase 5: Faithful explanation, 2 weeks.** Generate explanations from evidence JSON only: top scoring factors, constraint checks, and attribution values. Add automatic hallucination checks by verifying every explanation claim against evidence fields.

**Phase 6: Paper-grade evaluation, 3-5 weeks.** Run full baselines, ablations, significance tests, case studies, and efficiency analysis. Freeze prompts, model versions, seeds, and data splits.

## 7. Kill Tests

Do not position the paper as Q1-ready if any of these fail:

1. The external dataset cannot provide credible sequential user-recipe histories.
2. The proposed method does not beat SASRec/LightGCN-style baselines on NDCG@K or does so only by sacrificing health constraints.
3. Hard constraint satisfaction is not clearly better than soft-penalty baselines.
4. Explanations mention health or nutrition claims not present in evidence fields.
5. The study still depends only on synthetic users.

## 8. Target Venues

Best fit if the method and experiments are strong: IEEE TKDE, Information Fusion, Expert Systems with Applications, Knowledge-Based Systems, ACM TORS, Information Processing & Management. For a shorter applied version, aim at KDD Applied Data Science, WSDM, RecSys, or WWW Companion, but a一区 journal submission needs stronger experiments and a cleaner benchmark story.

