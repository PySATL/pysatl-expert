from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.stats import shapiro

from pysatl_expert.core.pipeline_components import PipelineComponents
from pysatl_expert.core.recommendation import ProcedureRequest
from pysatl_expert.criteria.criterion_adapter import CriterionAdapter
from pysatl_expert.criteria.scipy_adapter import ScipyShapiroAdapter
from pysatl_expert.models.analysis_result import ProcedureResult
from pysatl_expert.models.feature_extractor import FeatureExtractor
from pysatl_expert.pipeline import DistributionPipeline


def _empty_pipeline() -> DistributionPipeline:
    return DistributionPipeline(PipelineComponents([], None, None, FeatureExtractor()))


SAMPLE = np.array([0.1, 0.3, 0.8, 1.2, 1.7, 2.1])


def method(request):
    return SimpleNamespace(
        required_statistics=("normality",),
        procedure_requests={"normality": request},
        predict=lambda sample, analysis: analysis.procedure_results["normality"],
    )


def run(sample, request, *, enabled=False):
    return _empty_pipeline().recommend(
        sample,
        method=method(request),
        procedures={"normality": ScipyShapiroAdapter()},
        allow_calibration=enabled,
    )


def test_real_values_match_scipy_in_unified_result():
    result = run(SAMPLE, ProcedureRequest(p_value=True), enabled=True)
    reference = shapiro(SAMPLE)
    assert result.error is None
    assert result.prediction.statistic == pytest.approx(reference.statistic)
    assert result.prediction.p_value == pytest.approx(reference.pvalue)
    assert result.prediction.alternative == "less"
    assert result.prediction.source == "scipy.stats.shapiro"
    assert result.prediction.unavailable == {}


def test_same_consumer_accepts_scipy_and_criterion_adapter():
    criterion = MagicMock()
    criterion.alternative.return_value.type.return_value.value = "left"
    criterion.hypothesis.return_value.parameters.return_value = {}
    criterion.hypothesis.return_value.parameter_values = None
    criterion.configuration.return_value = {}
    criterion.code.return_value = "mock-criterion"
    criterion.execute_statistic.return_value = 0.8
    shared_method = method(ProcedureRequest())
    engine = _empty_pipeline()
    scipy_result = engine.recommend(
        SAMPLE, method=shared_method, procedures={"normality": ScipyShapiroAdapter()}
    )
    criterion_result = engine.recommend(
        SAMPLE, method=shared_method, procedures={"normality": CriterionAdapter(criterion)}
    )
    assert scipy_result.error is None and criterion_result.error is None
    assert scipy_result.prediction.source == "scipy.stats.shapiro"
    assert criterion_result.prediction.source == "mock-criterion"
    assert criterion_result.prediction.statistic == 0.8


def test_p_value_is_not_exposed_without_explicit_permission():
    result = run(SAMPLE, ProcedureRequest(p_value=True))
    assert result.error == "Required calibration unavailable"
    assert result.analysis.procedure_results["normality"].p_value is None
    assert result.analysis.procedure_results["normality"].unavailable == {
        "p_value": "calibration not enabled"
    }
    assert result.analysis.special_statistics["normality"] == pytest.approx(
        shapiro(SAMPLE).statistic
    )


def test_available_p_value_survives_unavailable_cv():
    request = ProcedureRequest(p_value=True, critical_values=True, required=False)
    result = run(SAMPLE, request, enabled=True)
    assert result.error is None
    assert result.prediction.p_value == pytest.approx(shapiro(SAMPLE).pvalue)
    assert "does not provide" in result.prediction.unavailable["critical_values"]
    assert "does not provide" in result.analysis.calibration_failures["normality"]


def test_required_missing_cv_prevents_method_call_and_retains_p_value():
    req = ProcedureRequest(p_value=True, critical_values=True)
    recommendation_method = method(req)
    recommendation_method.predict = MagicMock()
    result = _empty_pipeline().recommend(
        SAMPLE,
        method=recommendation_method,
        procedures={"normality": ScipyShapiroAdapter()},
        allow_calibration=True,
    )
    assert result.error == "Required calibration unavailable"
    assert result.analysis.procedure_results["normality"].p_value == pytest.approx(
        shapiro(SAMPLE).pvalue
    )
    recommendation_method.predict.assert_not_called()


def test_large_sample_preserves_w_but_withholds_unreliable_p_value():
    result = run(np.arange(5001), ProcedureRequest(p_value=True, required=False), enabled=True)
    assert result.error is None
    assert np.isfinite(result.prediction.statistic)
    assert result.prediction.p_value is None
    assert "N > 5000" in result.prediction.unavailable["p_value"]


def test_short_sample_records_procedure_failure():
    assert "at least 3" in run([1, 2], ProcedureRequest()).analysis.failures["normality"]


@pytest.mark.parametrize("sample", [[1, 1, 1], [1, 2, np.nan]])
def test_invalid_sample_is_rejected_before_procedure(sample):
    with pytest.raises(ValueError):
        run(sample, ProcedureRequest())


def test_unavailable_field_cannot_also_be_returned():
    with pytest.raises(ValueError, match="both available and unavailable"):
        ProcedureResult(
            statistic=0.9,
            alternative="less",
            p_value=0.1,
            calibration_method="test",
            unavailable={"p_value": "missing"},
        )
