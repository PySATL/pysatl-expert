import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from pysatl_expert.models.hierarchical_model import HierarchicalExpertModel


def _training_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    labels = np.repeat(["A", "B", "C", "D"], 8)
    class_centers = {"A": 0.0, "B": 1.0, "C": 3.0, "D": 4.0}
    values = np.array(
        [[class_centers[label] + rng.normal(0, 0.05), rng.normal(0, 0.05)] for label in labels]
    )
    return pd.DataFrame(values, columns=["shape", "aux"]), pd.Series(labels)


def test_hierarchical_model_fits_predicts_and_returns_normalized_probabilities():
    features, target = _training_data()
    model = HierarchicalExpertModel({"low": ["A", "B"], "high": ["C", "D"]})

    model.fit(features, target, n_stage1=None, n_stage2=None)
    probabilities = model.predict_proba(features)
    predictions = model.predict(features)

    assert probabilities.shape == (len(features), 4)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert set(predictions) <= {"A", "B", "C", "D"}


def test_hierarchical_model_supports_missing_raw_statistics() -> None:
    features, target = _training_data()
    features.loc[features.index[::3], "aux"] = np.nan
    model = HierarchicalExpertModel({"low": ["A", "B"], "high": ["C", "D"]})

    model.fit(features, target, n_stage1=None, n_stage2=None)
    probabilities = model.predict_proba(features.iloc[:2])

    assert probabilities.shape == (2, 4)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_hierarchical_model_rejects_duplicate_distribution_in_family_map():
    with pytest.raises(ValueError, match="more than one family"):
        HierarchicalExpertModel({"first": ["A", "B"], "second": ["B", "C"]})


def test_hierarchical_model_requires_family_map_to_match_training_targets():
    features, target = _training_data()
    model = HierarchicalExpertModel({"low": ["A", "B"], "high": ["C"]})

    with pytest.raises(ValueError, match="family map does not match"):
        model.fit(features, target)


def test_hierarchical_model_rejects_ndarray_with_wrong_feature_count():
    features, target = _training_data()
    model = HierarchicalExpertModel({"low": ["A", "B"], "high": ["C", "D"]})
    model.fit(features, target, n_stage1=None, n_stage2=None)

    with pytest.raises(ValueError, match="expected 2 features"):
        model.predict(np.zeros((1, 3)))


def test_hierarchical_model_never_selects_numerically_unstable_features():
    features, target = _training_data()
    features["normal__glb"] = np.tile([0.0, 1.0], len(features) // 2)
    model = HierarchicalExpertModel({"low": ["A", "B"], "high": ["C", "D"]})

    model.fit(features, target, n_stage1=None, n_stage2=None)

    assert "normal__glb" not in model.stage1_features
    assert all(
        "normal__glb" not in family_features for family_features in model.stage2_features.values()
    )
    assert model.feature_names == features.columns.tolist()


def test_hierarchical_model_uses_requested_forest_resources():
    features, target = _training_data()
    model = HierarchicalExpertModel({"low": ["A", "B"], "high": ["C", "D"]})

    model.fit(
        features,
        target,
        n_stage1=None,
        n_stage2=None,
        n_estimators=7,
        n_jobs=1,
    )

    assert model.stage1_model.n_estimators == 7
    assert model.stage1_model.n_jobs == 1
    assert all(forest.n_estimators == 7 for forest in model.stage2_models.values())
    assert all(forest.n_jobs == 1 for forest in model.stage2_models.values())


def test_hierarchical_model_trains_final_forests_on_explicit_features():
    features, target = _training_data()
    model = HierarchicalExpertModel({"low": ["A", "B"], "high": ["C", "D"]})

    model.fit_selected(
        features,
        target,
        stage1_features=["shape"],
        stage2_features={"low": ["aux"], "high": ["shape"]},
        n_estimators=7,
        n_jobs=1,
    )

    assert model.stage1_features == ["shape"]
    assert model.stage2_features == {"low": ["aux"], "high": ["shape"]}
    assert model.stage1_model.n_features_in_ == 1
    assert all(forest.n_features_in_ == 1 for forest in model.stage2_models.values())


def test_hierarchical_model_rejects_invalid_explicit_feature_selection():
    features, target = _training_data()
    model = HierarchicalExpertModel({"low": ["A", "B"], "high": ["C", "D"]})

    with pytest.raises(ValueError, match="missing Stage 2 selections"):
        model.fit_selected(
            features,
            target,
            stage1_features=["shape"],
            stage2_features={"low": ["aux"]},
        )

    with pytest.raises(ValueError, match="unknown feature"):
        model.fit_selected(
            features,
            target,
            stage1_features=["unknown"],
            stage2_features={"low": ["aux"], "high": ["shape"]},
        )


def test_feature_selection_supports_per_family_budgets_and_complete_stage2():
    features, target = _training_data()
    features["third"] = np.linspace(0.0, 1.0, len(features))
    features.loc[target == "A", "aux"] = np.nan
    model = HierarchicalExpertModel({"low": ["A", "B"], "high": ["C", "D"]})

    stage1, stage2 = model.select_features(
        features,
        target,
        n_stage1=2,
        n_stage2={"low": 1, "high": 2},
        complete_stage2=True,
        n_estimators=7,
        n_jobs=1,
    )

    assert len(stage1) == 2
    assert len(stage2["low"]) == 1
    assert len(stage2["high"]) == 2
    assert "aux" not in stage2["low"]


def test_batched_selection_importances_match_one_complete_forest():
    rng = np.random.RandomState(7)
    features = rng.normal(size=(1_000, 12)).astype(np.float32)
    target = (
        features[:, 0] + 0.7 * features[:, 3] + rng.normal(size=len(features)) * 0.2 > 0
    ).astype(int)
    complete_forest = RandomForestClassifier(
        n_estimators=37,
        max_depth=15,
        bootstrap=True,
        random_state=42,
        n_jobs=1,
    ).fit(features, target)

    batched_importances = HierarchicalExpertModel._batched_feature_importances(
        features,
        target,
        n_estimators=37,
        n_jobs=1,
        batch_size=7,
    )

    np.testing.assert_array_equal(
        batched_importances,
        complete_forest.feature_importances_,
    )
