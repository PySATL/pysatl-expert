import numpy as np
import pytest

from pysatl_expert.models.analysis_result import AnalysisResult
from pysatl_expert.models.feature_vector import FeatureVector
from pysatl_expert.models.model_manifest import (
    ModelCompatibilityError,
    write_model_manifest,
)
from pysatl_expert.strategy.ml_strategy import MLStrategy


def test_ml_strategy_verifies_manifest_before_loading_model(tmp_path, monkeypatch):
    class CompatibleModel:
        classes_ = np.array(["Normal", "Student"])
        feature_names = list(FeatureVector.FEATURE_NAMES)

    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"placeholder")
    calls = []
    monkeypatch.setattr(
        "pysatl_expert.strategy.ml_strategy.verify_model_manifest",
        lambda *_: calls.append("verify")
        or {
            "model": {"sha256": "a" * 64, "filename": "model.joblib"},
            "feature_schema": {
                "sha256": "b" * 64,
                "names": list(FeatureVector.FEATURE_NAMES),
            },
        },
    )
    monkeypatch.setattr(
        "pysatl_expert.strategy.ml_strategy.load_model",
        lambda _: calls.append("load") or CompatibleModel(),
    )
    monkeypatch.setattr(
        "pysatl_expert.strategy.ml_strategy.validate_loaded_model",
        lambda *_: calls.append("validate"),
    )

    strategy = MLStrategy(model_path)

    assert strategy._class_names == ["Normal", "Student"]
    assert strategy._artifact_identity == {
        "sha256": "a" * 64,
        "identifier": "model.joblib",
        "feature_schema": {
            "sha256": "b" * 64,
            "names": list(FeatureVector.FEATURE_NAMES),
        },
    }
    assert calls == ["verify", "load", "validate"]


def test_ml_strategy_does_not_deserialize_model_when_manifest_is_invalid(tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"placeholder")
    loaded = False

    def reject(*_):
        raise ModelCompatibilityError("Model SHA-256 mismatch")

    def load(_):
        nonlocal loaded
        loaded = True

    monkeypatch.setattr("pysatl_expert.strategy.ml_strategy.verify_model_manifest", reject)
    monkeypatch.setattr("pysatl_expert.strategy.ml_strategy.load_model", load)

    with pytest.raises(ModelCompatibilityError, match="SHA-256"):
        MLStrategy(model_path)

    assert loaded is False


def test_ml_strategy_rejects_criterion_revision_before_deserialization(tmp_path, monkeypatch):
    class CompatibleModel:
        feature_names = list(FeatureVector.FEATURE_NAMES)
        family_map = {
            "SymmetricUnbounded": ["Normal", "Student"],
            "PositiveUnbounded": ["Exponential", "Gamma", "LogNormal", "Weibull"],
            "Bounded": ["Beta", "Uniform"],
        }
        stage1_features = ["normal__ks", "skew"]
        stage2_features = {
            "SymmetricUnbounded": ["normal__ks"],
            "PositiveUnbounded": ["gamma__ks"],
            "Bounded": ["beta__ks"],
        }
        classes_ = np.array(sorted(member for members in family_map.values() for member in members))

    selection_document = {
        "schema_version": 1,
        "family_map": CompatibleModel.family_map,
        "feature_schema": list(FeatureVector.FEATURE_NAMES),
        "selector": {"stage1_feature_count": 2},
        "selection": {
            "stage1_features": CompatibleModel.stage1_features,
            "stage2_features": CompatibleModel.stage2_features,
        },
    }
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"placeholder")
    monkeypatch.setattr(
        "pysatl_expert.models.model_manifest._installed_criterion_revision",
        lambda: "a" * 40,
    )
    write_model_manifest(model_path, CompatibleModel(), selection_document)
    monkeypatch.setattr(
        "pysatl_expert.models.model_manifest._installed_criterion_revision",
        lambda: "b" * 40,
    )
    loaded = False

    def load(_):
        nonlocal loaded
        loaded = True

    monkeypatch.setattr("pysatl_expert.strategy.ml_strategy.load_model", load)

    with pytest.raises(ModelCompatibilityError, match="Criterion source revision is not validated"):
        MLStrategy(model_path)

    assert loaded is False


def test_ml_strategy_uses_the_model_bundle_feature_schema(tmp_path, monkeypatch):
    class BundledModel:
        classes_ = np.array(["Normal"])
        feature_names = ["sample_size", "normal__ks"]

    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"placeholder")
    monkeypatch.setattr(
        "pysatl_expert.strategy.ml_strategy.verify_model_manifest",
        lambda *_: {
            "model": {"sha256": "a" * 64, "filename": "model.joblib"},
            "feature_schema": {
                "sha256": "b" * 64,
                "names": BundledModel.feature_names,
            },
        },
    )
    monkeypatch.setattr("pysatl_expert.strategy.ml_strategy.validate_loaded_model", lambda *_: None)
    monkeypatch.setattr("pysatl_expert.strategy.ml_strategy.load_model", lambda _: BundledModel())

    strategy = MLStrategy(model_path)

    assert strategy.feature_names == BundledModel.feature_names


def test_predict_preserves_legacy_nan_projection_and_class_scores():
    class NaNAwareModel:
        def predict_proba(self, features):
            assert features.shape == (1, 2)
            assert np.isnan(features[0, 1])
            return np.array([[0.2, 0.8]])

    strategy = object.__new__(MLStrategy)
    strategy.model = NaNAwareModel()
    strategy._feature_names = ["sample_size", "normal__ks"]
    strategy._class_names = ["Normal", "Uniform"]
    analysis = AnalysisResult(
        {"sample_size": 80, "mean": 0.5},
        {},
        {"Normal": {}},
        {"normal__ks": "criterion unavailable"},
    )

    prediction = strategy.predict(np.arange(80), analysis)

    assert prediction == {
        "label": "Uniform",
        "scores": {"Normal": 0.2, "Uniform": 0.8},
        "evidence": {},
        "artifact": {},
    }


def test_recommendation_adapter_uses_manifest_order_and_runs_inference_once():
    class CountingModel:
        calls = 0

        def predict_proba(self, features):
            self.calls += 1
            assert features.shape == (1, len(FeatureVector.FEATURE_NAMES))
            assert features[0, FeatureVector.FEATURE_NAMES.index("normal__ks")] == 0.5
            return np.array([[0.75, 0.25]])

    strategy = object.__new__(MLStrategy)
    strategy.model = CountingModel()
    strategy._feature_names = list(FeatureVector.FEATURE_NAMES)
    strategy._class_names = ["Normal", "Student"]
    analysis = AnalysisResult(
        {"sample_size": 100},
        {},
        {"Normal": {"ks": 0.5}},
        {},
    )

    prediction = strategy.predict(np.arange(100), analysis)

    assert prediction == {
        "label": "Normal",
        "scores": {"Normal": 0.75, "Student": 0.25},
        "evidence": {},
        "artifact": {},
    }


def test_ml_strategy_rejects_absent_catalog_but_keeps_individual_nan_features():
    class NaNAwareModel:
        calls = 0

        def predict_proba(self, features):
            self.calls += 1
            assert np.isnan(features[0, 1])
            return np.array([[1.0]])

    strategy = object.__new__(MLStrategy)
    strategy.model = NaNAwareModel()
    strategy._feature_names = ["sample_size", "normal__ks"]
    strategy._class_names = ["Normal"]

    empty = AnalysisResult({"sample_size": 80}, {}, {}, {})
    with pytest.raises(ValueError, match="Catalog analysis has no statistic results"):
        strategy.predict(np.arange(80), empty)

    partial = AnalysisResult({"sample_size": 80}, {}, {"Normal": {}}, {"normal__ks": "unavailable"})
    result = strategy.predict(np.arange(80), partial)
    assert result["label"] == "Normal"
    assert result["scores"] == {"Normal": 1.0}
    assert strategy.model.calls == 1
