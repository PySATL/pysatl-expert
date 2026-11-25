from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.stats import shapiro
from sklearn.linear_model import LogisticRegression

from pysatl_expert.core.pipeline_components import PipelineComponents
from pysatl_expert.models.feature_extractor import FeatureExtractor
from pysatl_expert.pipeline import DistributionPipeline


def pipeline():
    distribution = MagicMock()
    distribution.name = "Unused"
    distribution.support = (-np.inf, np.inf)
    distribution.fit.side_effect = AssertionError("unrequested fit")
    return DistributionPipeline(
        PipelineComponents([distribution], MagicMock(), None, FeatureExtractor())
    )


def test_real_statistic_feeds_real_second_model():
    # A deliberately tiny fitted model tests integration only, not scientific quality.
    estimator = LogisticRegression().fit([[0.0], [0.1], [0.9], [1.0]], [0, 0, 1, 1])

    class ShapiroStatistic:
        def calculate(self, sample):
            return shapiro(sample).statistic

    class Method:
        required_statistics = ("shapiro",)

        def predict(self, sample, analysis):
            assert len(sample) == analysis.sample_stats["sample_size"]
            return int(estimator.predict([[analysis.special_statistics["shapiro"]]])[0])

    sample = [0.1, 0.3, 0.8, 1.2, 1.7, 2.1]
    result = pipeline().recommend(
        sample, method=Method(), procedures={"shapiro": ShapiroStatistic()}
    )
    assert result.error is None
    assert result.analysis.special_statistics["shapiro"] == pytest.approx(shapiro(sample).statistic)
    assert result.prediction == estimator.predict([[shapiro(sample).statistic]])[0]


def test_selected_procedure_changes_decision_and_unselected_is_not_called():
    method = SimpleNamespace(
        required_statistics=("special",),
        predict=lambda sample, analysis: analysis.special_statistics["special"] > 0.5,
    )
    unused = MagicMock()
    for value, expected in [(0.1, False), (0.9, True)]:
        procedure = SimpleNamespace(calculate=lambda sample, value=value: value)
        result = pipeline().recommend(
            [1, 2, 3], method=method, procedures={"special": procedure, "unused": unused}
        )
        assert result.prediction is expected
    unused.calculate.assert_not_called()


def test_fitted_criterion_can_be_provided_as_procedure():
    from pysatl_expert.criteria.calculate.generic import GenericCriterion
    from pysatl_expert.criteria.catalog import CRITERIA_SPEC_BY_KEY
    from pysatl_expert.distributions.normal import NormalDistribution

    class NormalKs:
        def calculate(self, sample):
            distribution = NormalDistribution()
            statistic_class = CRITERIA_SPEC_BY_KEY[("normal", "ks")].statistic_class
            params = distribution.fit(sample)
            return GenericCriterion(statistic_class).calculate(sample, distribution, params)

    method = SimpleNamespace(
        required_statistics=("normal__ks",),
        predict=lambda sample, analysis: analysis.special_statistics["normal__ks"],
    )
    result = pipeline().recommend(
        [0.1, 0.4, 0.8, 1.2, 1.5], method=method, procedures={"normal__ks": NormalKs()}
    )
    assert result.error is None
    assert result.prediction == pytest.approx(0.1836162096661077)


def test_failure_preserves_success_and_prevents_method_execution():
    def fail(sample):
        raise ValueError("failed statistic")

    method = SimpleNamespace(required_statistics=("ok", "bad"), predict=MagicMock())
    result = pipeline().recommend(
        [1, 2, 3],
        method=method,
        procedures={
            "ok": SimpleNamespace(calculate=lambda sample: 1.0),
            "bad": SimpleNamespace(calculate=fail),
        },
    )
    assert result.analysis.special_statistics == {"ok": 1.0}
    assert "failed statistic" in result.analysis.failures["bad"]
    assert result.error == "Required statistics unavailable"
    method.predict.assert_not_called()


def test_missing_procedure_is_rejected_before_calculation():
    procedure = MagicMock()
    method = SimpleNamespace(required_statistics=("available", "missing"))
    with pytest.raises(ValueError, match="missing"):
        pipeline().recommend([1, 2, 3], method=method, procedures={"available": procedure})
    procedure.calculate.assert_not_called()


def test_method_failure_preserves_analysis():
    def fail(sample, analysis):
        raise RuntimeError("prediction failed")

    method = SimpleNamespace(required_statistics=("s",), predict=fail)
    result = pipeline().recommend(
        [1, 2, 3], method=method, procedures={"s": SimpleNamespace(calculate=lambda sample: 2)}
    )
    assert result.analysis.special_statistics == {"s": 2.0}
    assert "prediction failed" in result.error


def test_nonfinite_statistic_is_unavailable():
    method = SimpleNamespace(required_statistics=("s",), predict=MagicMock())
    result = pipeline().recommend(
        [1, 2, 3],
        method=method,
        procedures={"s": SimpleNamespace(calculate=lambda sample: np.nan)},
    )
    assert "not finite" in result.analysis.failures["s"]
    method.predict.assert_not_called()
