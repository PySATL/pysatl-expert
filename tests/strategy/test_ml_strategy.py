import numpy as np
import pytest

from pysatl_expert.models.feature_vector import FeatureVector
from pysatl_expert.models.model_manifest import ModelCompatibilityError
from pysatl_expert.strategy.ml_strategy import MLStrategy


class PredictingModel:
    def predict_proba(self, features):
        assert features.shape == (1, len(FeatureVector.FEATURE_NAMES))
        ks_index = FeatureVector.FEATURE_NAMES.index("normal__ks")
        assert features[0, ks_index] == 0.5
        return np.array([[0.75, 0.25]])


class BootstrapVotingModel(PredictingModel):
    def __init__(self):
        self.bootstrap_predictions = iter(["Student", "Student", "Normal"])

    def predict(self, features):
        assert features.shape == (1, len(FeatureVector.FEATURE_NAMES))
        return np.array([next(self.bootstrap_predictions)])


def test_predict_report_separates_gof_scores_from_class_probabilities():
    strategy = object.__new__(MLStrategy)
    strategy.model = PredictingModel()
    strategy._class_names = ["Normal", "Student"]
    feature_vector = FeatureVector(
        {"sample_size": 100},
        {"Normal": {"ks": 0.5}},
    )

    report = strategy.predict_report(feature_vector)

    assert report.final_ranks == {"Normal": 0.75, "Student": 0.25}
    assert report.confidence_kind == "model_probability"
    assert report.model_confidence == 0.75
    assert report.bootstrap_stability is None
    assert report.all_scores["Normal"]["ks"] == 0.5
    assert np.isnan(report.all_scores["Student"]["ks"])
    assert report.sample_statistics == {"sample_size": 100}


def test_predict_report_keeps_base_ranking_when_bootstrap_prefers_another_class():
    strategy = object.__new__(MLStrategy)
    strategy.model = BootstrapVotingModel()
    strategy._class_names = ["Normal", "Student"]
    feature_vector = FeatureVector(
        {"sample_size": 100},
        {"Normal": {"ks": 0.5}},
    )

    report = strategy.predict_report(feature_vector, [feature_vector] * 3)

    assert report.distribution_name == "Normal"
    assert report.confidence == 0.75
    assert report.confidence_kind == "model_probability"
    assert report.bootstrap_stability == 0.333
    assert report.model_ranks == {"Normal": 0.75, "Student": 0.25}
    assert report.model_confidence == 0.75
    assert report.final_ranks == {"Normal": 0.75, "Student": 0.25}
    assert report.bootstrap_ranks == pytest.approx({"Normal": 1 / 3, "Student": 2 / 3})
    assert report.bootstrap_successful == 3


@pytest.mark.parametrize("bootstrap_fvs", [None, []])
def test_predict_report_without_bootstrap_has_no_stability(bootstrap_fvs):
    strategy = object.__new__(MLStrategy)
    strategy.model = PredictingModel()
    strategy._class_names = ["Normal", "Student"]
    fv = FeatureVector({"sample_size": 100}, {"Normal": {"ks": 0.5}})

    report = strategy.predict_report(fv, bootstrap_fvs)

    assert report.bootstrap_stability is None
    assert report.bootstrap_ranks == {}
    assert report.bootstrap_successful == 0


def test_zero_bootstrap_agreement_is_not_missing_stability():
    strategy = object.__new__(MLStrategy)
    strategy.model = BootstrapVotingModel()
    strategy.model.bootstrap_predictions = iter(["Student"] * 3)
    strategy._class_names = ["Normal", "Student"]
    fv = FeatureVector({"sample_size": 100}, {"Normal": {"ks": 0.5}})

    report = strategy.predict_report(fv, [fv] * 3)

    assert report.distribution_name == "Normal"
    assert report.bootstrap_stability == 0.0
    assert report.bootstrap_ranks == {"Normal": 0.0, "Student": 1.0}
    assert report.confidence == 0.75


def test_unknown_bootstrap_class_is_not_silently_dropped():
    strategy = object.__new__(MLStrategy)
    strategy.model = BootstrapVotingModel()
    strategy.model.bootstrap_predictions = iter(["Unknown"])
    strategy._class_names = ["Normal", "Student"]
    fv = FeatureVector({"sample_size": 100}, {"Normal": {"ks": 0.5}})

    with pytest.raises(ValueError, match="unknown class"):
        strategy.predict_report(fv, [fv])


def test_predict_report_has_no_sample_size_constraint():
    strategy = object.__new__(MLStrategy)
    strategy.model = PredictingModel()
    strategy._class_names = ["Normal", "Student"]
    feature_vector = FeatureVector(
        {"sample_size": 49},
        {"Normal": {"ks": 0.5}},
    )

    report = strategy.predict_report(feature_vector)

    assert report.distribution_name == "Normal"


def test_ml_strategy_verifies_manifest_before_loading_model(tmp_path, monkeypatch):
    class CompatibleModel:
        classes_ = np.array(["Normal", "Student"])
        feature_names = FeatureVector.FEATURE_NAMES

    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"placeholder")
    calls = []
    monkeypatch.setattr(
        "pysatl_expert.strategy.ml_strategy.verify_model_manifest",
        lambda *_: calls.append("verify")
        or {"feature_schema": {"names": FeatureVector.FEATURE_NAMES}},
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


def test_ml_strategy_uses_the_model_bundle_feature_schema(tmp_path, monkeypatch):
    class BundledModel:
        classes_ = np.array(["Normal"])
        feature_names = ["sample_size", "normal__ks"]

    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"placeholder")
    monkeypatch.setattr(
        "pysatl_expert.strategy.ml_strategy.verify_model_manifest",
        lambda *_: {"feature_schema": {"names": BundledModel.feature_names}},
    )
    monkeypatch.setattr("pysatl_expert.strategy.ml_strategy.validate_loaded_model", lambda *_: None)
    monkeypatch.setattr("pysatl_expert.strategy.ml_strategy.load_model", lambda _: BundledModel())

    strategy = MLStrategy(model_path)

    assert strategy.feature_names == BundledModel.feature_names
