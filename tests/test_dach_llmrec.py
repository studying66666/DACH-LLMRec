import builtins

from dach_llmrec import DACHLLMRecommender
from dach_llmrec.bpr import train_bpr
from dach_llmrec.demo_data import create_demo_database
from dach_llmrec.diagnostics import diagnose_bpr
from dach_llmrec.evaluate import evaluate
from dach_llmrec.embeddings import HashEmbeddingProvider, SentenceTransformerEmbeddingProvider
from dach_llmrec.experiments.run_all import run_all
from dach_llmrec.llmrec_aug import build_llmrec_augmented_edges
from dach_llmrec.weight_search import grid_search_recipe_weights



class _FakeEmbeddingVector:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class _FakeSentenceTransformer:
    def __init__(self, values=(3.0, 4.0, 0.0)):
        self.values = list(values)
        self.calls = 0

    def encode(self, text, normalize_embeddings=True, convert_to_numpy=True):
        self.calls += 1
        return _FakeEmbeddingVector(self.values)


class _StaticEmbeddingProvider:
    def embed(self, text: str) -> list[float]:
        base = sum(ord(char) for char in text) % 17
        return [float(base + 1), 1.0, 0.5]


def _patch_real_embedding_factory(monkeypatch):
    def fake_build_embedding_provider(
        embedding_provider="hash",
        embedding_model="BAAI/bge-small-zh-v1.5",
        embedding_device="auto",
        embedding_cache_dir=None,
    ):
        if embedding_provider == "real":
            return _StaticEmbeddingProvider()
        return HashEmbeddingProvider()

    monkeypatch.setattr(
        "dach_llmrec.evaluate.build_embedding_provider",
        fake_build_embedding_provider,
    )
def _demo_db(tmp_path):
    return create_demo_database(tmp_path / "demo.sqlite")


def test_sentence_transformer_embedding_provider_returns_vector(tmp_path):
    fake_model = _FakeSentenceTransformer()
    provider = SentenceTransformerEmbeddingProvider(
        model_name_or_path="fake-zh-model",
        device="cpu",
        cache_dir=tmp_path / "cache",
        model=fake_model,
    )

    vector = provider.embed("tomato egg recipe")

    assert isinstance(vector, list)
    assert vector == [0.6, 0.8, 0.0]
    assert fake_model.calls == 1


def test_sentence_transformer_embedding_provider_uses_disk_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    first_model = _FakeSentenceTransformer((1.0, 0.0, 0.0))
    first_provider = SentenceTransformerEmbeddingProvider(
        model_name_or_path="fake-zh-model",
        device="cpu",
        cache_dir=cache_dir,
        model=first_model,
    )
    assert first_provider.embed("cache text") == [1.0, 0.0, 0.0]

    second_model = _FakeSentenceTransformer((0.0, 1.0, 0.0))
    second_provider = SentenceTransformerEmbeddingProvider(
        model_name_or_path="fake-zh-model",
        device="cpu",
        cache_dir=cache_dir,
        model=second_model,
    )

    assert second_provider.embed("cache text") == [1.0, 0.0, 0.0]
    assert second_model.calls == 0


def test_sentence_transformer_embedding_provider_missing_dependency_is_clear(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("missing sentence_transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    try:
        SentenceTransformerEmbeddingProvider(
            model_name_or_path="fake-zh-model",
            device="cpu",
            cache_dir=None,
        )
    except RuntimeError as exc:
        assert "Could not import sentence-transformers" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for missing sentence-transformers")

def test_recipe_recommendations_pass_hard_filters(tmp_path):
    db_path = _demo_db(tmp_path)
    recommender = DACHLLMRecommender(db_path)
    try:
        result = recommender.recommend(user_id=1, top_k=3, mode="recipe")
        assert result["mode"] == "recipe"
        assert len(result["items"]) == 3
        validation = recommender.validate(user_id=1, top_k=3)
        assert validation["passed"], validation["violations"]
    finally:
        recommender.close()


def test_ingredient_recommendations_return_expected_shape(tmp_path):
    db_path = _demo_db(tmp_path)
    recommender = DACHLLMRecommender(db_path)
    try:
        result = recommender.recommend(user_id=1, top_k=5, mode="ingredient")
        assert result["mode"] == "ingredient"
        assert len(result["items"]) == 5
        for item in result["items"]:
            assert item["item_type"] == "ingredient"
            assert "evidence" in item
            assert "explanation" in item
    finally:
        recommender.close()


def test_disease_constraints_are_opt_in(tmp_path):
    db_path = _demo_db(tmp_path)
    recommender = DACHLLMRecommender(db_path)
    try:
        base = recommender.recommend(user_id=1, top_k=3)
        disease = recommender.recommend(
            user_id=1,
            top_k=3,
            health_factors=[{"type": "disease", "id": 22}],
            enable_disease_constraints=True,
        )
        assert not base["metadata"]["disease_constraints_enabled"]
        assert disease["metadata"]["disease_constraints_enabled"]
    finally:
        recommender.close()


def test_bpr_training_artifact_can_be_loaded(tmp_path):
    db_path = _demo_db(tmp_path)
    model_path = tmp_path / "smoke_bpr.pt"
    summary = train_bpr(db_path, output=model_path, epochs=1, dim=8, batch_size=1024, device="cpu")
    assert summary["device"] == "cpu"
    assert model_path.exists()

    recommender = DACHLLMRecommender(db_path, bpr_model_path=model_path)
    try:
        result = recommender.recommend(user_id=1, top_k=3, mode="recipe")
        assert len(result["items"]) == 3
    finally:
        recommender.close()


def test_bpr_diagnostics_report_train_test_coverage(tmp_path):
    db_path = _demo_db(tmp_path)
    model_path = tmp_path / "diagnostic_bpr.pt"
    train_bpr(db_path, output=model_path, epochs=1, dim=8, batch_size=8, device="cpu")

    result = diagnose_bpr(
        db_path=db_path,
        cutoff="2026-06-01 00:00:00",
        top_k=3,
        max_users=3,
        bpr_model_path=model_path,
    )

    assert result["training_split"]["users_with_positive_history"] == 2
    assert result["training_split"]["negative_sampling"]["negative_samples_per_positive"] == 2
    assert result["training_split"]["negative_sampling"]["random_negative_ratio"] >= 0.0
    assert result["evaluation_split"]["evaluated_users"] >= 2
    assert result["evaluation_split"]["users_with_training_history"] >= 1
    assert result["evaluation_split"]["test_positive_candidate_coverage"] >= 1.0
    assert not result["bpr_top_k"]["skipped"]
    assert "history_overlap_rate" in result["bpr_top_k"]

def test_itemknn_evaluation_ranker_runs(tmp_path):
    db_path = _demo_db(tmp_path)

    result = evaluate(
        db_path=db_path,
        cutoff="2026-06-01 00:00:00",
        top_k=3,
        max_users=3,
        rankers=["itemknn"],
    )

    assert result["metadata"]["evaluated_users"] > 0
    assert "itemknn" in result["results"]
    assert result["results"]["itemknn"]["coverage"] >= 0.0
    assert result["results"]["itemknn"]["safety_violation_rate"] == 0.0

def test_content_feedback_hybrid_ranker_runs(tmp_path):
    db_path = _demo_db(tmp_path)

    result = evaluate(
        db_path=db_path,
        cutoff="2026-06-01 00:00:00",
        top_k=3,
        max_users=3,
        rankers=["content_feedback"],
    )

    assert result["metadata"]["evaluated_users"] > 0
    assert "content_feedback" in result["results"]
    assert result["results"]["content_feedback"]["coverage"] >= 0.0
    assert result["results"]["content_feedback"]["safety_violation_rate"] == 0.0

def test_als_evaluation_ranker_runs(tmp_path):
    db_path = _demo_db(tmp_path)

    result = evaluate(
        db_path=db_path,
        cutoff="2026-06-01 00:00:00",
        top_k=3,
        max_users=3,
        rankers=["als_only"],
    )

    assert result["metadata"]["evaluated_users"] > 0
    assert "als_only" in result["results"]
    assert result["results"]["als_only"]["coverage"] >= 0.0
    assert result["results"]["als_only"]["safety_violation_rate"] == 0.0

def test_fusion_lr_evaluation_ranker_runs(tmp_path):
    db_path = _demo_db(tmp_path)
    model_path = tmp_path / "fusion_bpr.pt"
    train_bpr(db_path, output=model_path, epochs=1, dim=8, batch_size=8, device="cpu")

    result = evaluate(
        db_path=db_path,
        cutoff="2026-06-01 00:00:00",
        top_k=3,
        max_users=3,
        bpr_model_path=model_path,
        rankers=["fusion_lr"],
    )

    assert result["metadata"]["evaluated_users"] > 0
    assert "fusion_lr" in result["results"]
    assert not result["results"]["fusion_lr"].get("skipped")
    assert result["results"]["fusion_lr"]["coverage"] >= 0.0
    assert result["results"]["fusion_lr"]["safety_violation_rate"] == 0.0
    assert result["models"]["fusion_lr"]["samples"] > 0


def test_grid_search_weight_optimizer_reports_validation_ndcg(tmp_path):
    db_path = _demo_db(tmp_path)

    result = grid_search_recipe_weights(
        db_path=db_path,
        cutoff="2026-06-01 00:00:00",
        top_k=3,
        max_users=3,
        grid_step=0.5,
        max_component_weight=1.0,
    )

    summary = result["grid_search"]
    assert summary["selection_metric"] == "ndcg@3"
    assert summary["validation_users"] > 0
    assert summary["candidate_weight_count"] > 0
    assert summary["best_validation_ndcg_at_k"] >= 0.0
    assert set(summary["best_weights"]) == {
        "preference",
        "health_goal",
        "content",
        "feedback",
        "llm_alignment",
        "quality",
        "diversity",
    }


def test_dach_grid_evaluation_ranker_runs(tmp_path):
    db_path = _demo_db(tmp_path)

    result = evaluate(
        db_path=db_path,
        cutoff="2026-06-01 00:00:00",
        top_k=3,
        max_users=3,
        rankers=["dach_grid"],
    )

    assert result["metadata"]["evaluated_users"] > 0
    assert "dach_grid" in result["results"]
    assert not result["results"]["dach_grid"].get("skipped")
    assert result["results"]["dach_grid"]["ndcg_at_k"] >= 0.0
    assert result["results"]["dach_grid"]["safety_violation_rate"] == 0.0
    assert result["models"]["dach_grid"]["best_validation_ndcg_at_k"] >= 0.0



def test_embedding_ablation_rankers_run(tmp_path, monkeypatch):
    _patch_real_embedding_factory(monkeypatch)
    db_path = _demo_db(tmp_path)

    result = evaluate(
        db_path=db_path,
        cutoff="2026-06-01 00:00:00",
        top_k=3,
        max_users=3,
        rankers=["dach_no_semantic", "dach_hash_embedding", "dach_real_embedding"],
        embedding_provider="real",
    )

    assert result["metadata"]["embedding_config"]["provider"] == "real"
    for ranker in ["dach_no_semantic", "dach_hash_embedding", "dach_real_embedding"]:
        assert ranker in result["results"]
        assert not result["results"][ranker].get("skipped")
        assert result["results"][ranker]["ndcg_at_k"] >= 0.0
        assert result["results"][ranker]["recall_at_k"] >= 0.0
        assert result["results"][ranker]["safety_violation_rate"] == 0.0

def test_llmrec_augmented_edges_are_generated(tmp_path):
    db_path = _demo_db(tmp_path)
    edges_path = tmp_path / "augmented_edges.json"

    result = build_llmrec_augmented_edges(
        db_path=db_path,
        output=edges_path,
        cutoff="2026-06-01 00:00:00",
        top_k=2,
        max_users=3,
        min_confidence=0.0,
    )

    assert edges_path.exists()
    assert result["metadata"]["edge_count"] == len(result["edges"])
    assert result["metadata"]["edge_count"] > 0
    assert result["edges"][0]["source"] == "evidence_constrained_llmrec_aug"
    assert "evidence" in result["edges"][0]


def test_bpr_training_uses_augmented_edges(tmp_path):
    db_path = _demo_db(tmp_path)
    edges_path = tmp_path / "augmented_edges.json"
    model_path = tmp_path / "augmented_bpr.pt"
    build_llmrec_augmented_edges(
        db_path=db_path,
        output=edges_path,
        cutoff="2026-06-01 00:00:00",
        top_k=2,
        max_users=3,
        min_confidence=0.0,
    )

    summary = train_bpr(
        db_path=db_path,
        output=model_path,
        cutoff="2026-06-01 00:00:00",
        epochs=1,
        dim=8,
        batch_size=8,
        device="cpu",
        augmented_edges_path=edges_path,
    )

    assert model_path.exists()
    assert summary["augmented_edges"]["enabled"]
    assert summary["augmented_edges"]["loaded_edges"] > 0
    assert summary["augmented_edges"]["added_positive_edges"] > 0


def test_llmrec_aug_bpr_evaluation_ranker_runs(tmp_path):
    db_path = _demo_db(tmp_path)
    edges_path = tmp_path / "augmented_edges.json"
    model_path = tmp_path / "augmented_bpr.pt"
    build_llmrec_augmented_edges(
        db_path=db_path,
        output=edges_path,
        cutoff="2026-06-01 00:00:00",
        top_k=2,
        max_users=3,
        min_confidence=0.0,
    )
    train_bpr(
        db_path=db_path,
        output=model_path,
        cutoff="2026-06-01 00:00:00",
        epochs=1,
        dim=8,
        batch_size=8,
        device="cpu",
        augmented_edges_path=edges_path,
    )

    result = evaluate(
        db_path=db_path,
        cutoff="2026-06-01 00:00:00",
        top_k=3,
        max_users=3,
        augmented_bpr_model_path=model_path,
        rankers=["llmrec_aug_bpr"],
    )

    assert result["metadata"]["evaluated_users"] > 0
    assert "llmrec_aug_bpr" in result["results"]
    assert not result["results"]["llmrec_aug_bpr"].get("skipped")
    assert result["results"]["llmrec_aug_bpr"]["ndcg_at_k"] >= 0.0
    assert result["results"]["llmrec_aug_bpr"]["safety_violation_rate"] == 0.0

def test_run_all_demo_experiment_writes_outputs(tmp_path, monkeypatch):
    _patch_real_embedding_factory(monkeypatch)
    output_dir = tmp_path / "experiment"
    result = run_all(
        db_path=None,
        output_dir=output_dir,
        use_demo_data=True,
        device="cpu",
        bpr_epochs=1,
        bpr_dim=8,
        bpr_batch_size=8,
        top_k=3,
        max_users=3,
    )
    assert (output_dir / "experiment.json").exists()
    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "diagnostics.json").exists()
    assert (output_dir / "weight_search.json").exists()
    assert (output_dir / "llmrec_augmented_edges.json").exists()
    assert (output_dir / "augmented_bpr_summary.json").exists()
    assert (output_dir / "embedding_ablation.json").exists()
    assert "dach_full" in result["evaluation"]["results"]
    assert "content" in result["evaluation"]["results"]
    assert "itemknn" in result["evaluation"]["results"]
    assert "als_only" in result["evaluation"]["results"]
    assert "llmrec_aug_bpr" in result["evaluation"]["results"]
    assert "fusion_lr" in result["evaluation"]["results"]
    assert "dach_grid" in result["evaluation"]["results"]
    assert "dach_no_semantic" in result["embedding_ablation"]
    assert "dach_hash_embedding" in result["embedding_ablation"]
    assert "dach_real_embedding" in result["embedding_ablation"]
    assert "diagnostics" in result
    assert result["augmentation_summary"]["edge_count"] > 0
    assert result["augmented_bpr_summary"]["augmented_edges"]["enabled"]
    assert result["weight_search"]["best_validation_ndcg_at_k"] >= 0.0
