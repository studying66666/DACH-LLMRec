from dach_llmrec import DACHLLMRecommender
from dach_llmrec.bpr import train_bpr
from dach_llmrec.demo_data import create_demo_database
from dach_llmrec.diagnostics import diagnose_bpr
from dach_llmrec.experiments.run_all import run_all


def _demo_db(tmp_path):
    return create_demo_database(tmp_path / "demo.sqlite")


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
    assert result["evaluation_split"]["evaluated_users"] == 2
    assert result["evaluation_split"]["users_with_training_history"] == 1
    assert result["evaluation_split"]["test_positive_candidate_coverage"] == 1.0
    assert not result["bpr_top_k"]["skipped"]
    assert "history_overlap_rate" in result["bpr_top_k"]

def test_run_all_demo_experiment_writes_outputs(tmp_path):
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
    assert "dach_full" in result["evaluation"]["results"]
    assert "content" in result["evaluation"]["results"]

