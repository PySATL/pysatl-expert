"""Regression coverage for seeded procedure evidence completeness."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pysatl_expert.core.pipeline_components import PipelineComponents
from pysatl_expert.core.recommendation import ProcedureRequest
from pysatl_expert.models.analysis_result import AnalysisResult, ProcedureResult
from pysatl_expert.models.feature_extractor import FeatureExtractor
from pysatl_expert.pipeline import DistributionPipeline


def pipeline():
    return DistributionPipeline(PipelineComponents([], MagicMock(), None, FeatureExtractor()))


def seeded_analysis():
    return AnalysisResult(
        sample_stats={"sample_size": 3},
        parameters={},
        statistics={},
        failures={},
        procedure_results={"s": ProcedureResult(statistic=0.7, source="seeded raw value")},
        sample_snapshot=(1.0, 2.0, 3.0),
    )


def setup_method(request):
    method = SimpleNamespace(
        required_statistics=("s",),
        procedure_requests={"s": request},
        predict=MagicMock(side_effect=lambda sample, analysis: analysis.procedure_results["s"]),
    )
    adapter = MagicMock()
    adapter.evaluate.return_value = ProcedureResult(
        statistic=0.7,
        alternative="greater",
        unavailable={"p_value": "calibration not enabled"},
        calibration_error="calibration not enabled",
        source="seed-regression",
    )
    return method, adapter


def test_seeded_raw_statistic_does_not_bypass_required_p_value():
    method, adapter = setup_method(ProcedureRequest(p_value=True))
    result = pipeline().recommend(
        [1, 2, 3],
        method=method,
        procedures={"s": adapter},
        analysis_provider=lambda sample: seeded_analysis(),
        allow_calibration=False,
    )
    assert adapter.evaluate.call_count == 1
    assert result.analysis.special_statistics["s"] == 0.7
    assert result.analysis.procedure_results["s"].p_value is None
    assert result.error == "Required calibration unavailable"
    method.predict.assert_not_called()


def test_seeded_optional_unavailable_p_value_allows_prediction():
    method, adapter = setup_method(ProcedureRequest(p_value=True, required=False))
    result = pipeline().recommend(
        [1, 2, 3],
        method=method,
        procedures={"s": adapter},
        analysis_provider=lambda sample: seeded_analysis(),
        allow_calibration=False,
    )
    assert adapter.evaluate.call_count == 1
    assert result.error is None
    method.predict.assert_called_once()


def test_complete_seeded_procedure_evidence_is_reused():
    method, adapter = setup_method(ProcedureRequest(p_value=True))
    seed = seeded_analysis()
    seed.procedure_results["s"] = ProcedureResult(
        statistic=0.7,
        alternative="greater",
        p_value=0.04,
        source="seed",
        calibration_method="explicit seed",
    )
    result = pipeline().recommend(
        [1, 2, 3],
        method=method,
        procedures={"s": adapter},
        analysis_provider=lambda sample: seed,
        allow_calibration=False,
    )
    adapter.evaluate.assert_not_called()
    assert result.error is None
    assert result.prediction.p_value == 0.04


def test_seeded_unavailable_required_fact_becomes_explicit_failure():
    method, adapter = setup_method(ProcedureRequest(p_value=True))
    seed = seeded_analysis()
    seed.procedure_results["s"] = ProcedureResult(
        statistic=0.7,
        alternative="greater",
        unavailable={"p_value": "calibration disabled upstream"},
        source="seed",
    )
    result = pipeline().recommend(
        [1, 2, 3],
        method=method,
        procedures={"s": adapter},
        analysis_provider=lambda sample: seed,
    )
    adapter.evaluate.assert_not_called()
    assert result.analysis.calibration_failures["s"] == "calibration disabled upstream"
    assert result.error == "Required calibration unavailable"
    method.predict.assert_not_called()


def test_seeded_calibration_at_another_alpha_is_recomputed():
    request = ProcedureRequest(critical_values=True, alpha=0.05)
    method, adapter = setup_method(request)
    seed = seeded_analysis()
    seed.procedure_results["s"] = ProcedureResult(
        statistic=0.7,
        alternative="greater",
        p_value=0.04,
        critical_values=(1.0,),
        alpha=0.1,
        source="seed",
        calibration_method="seed",
    )
    result = pipeline().recommend(
        [1, 2, 3],
        method=method,
        procedures={"s": adapter},
        analysis_provider=lambda sample: seed,
        allow_calibration=False,
    )
    adapter.evaluate.assert_called_once()
    assert result.error == "Required calibration unavailable"


def test_analysis_views_are_detached_from_procedure_results():
    analysis = seeded_analysis()
    raw_view = analysis.special_statistics
    raw_view.clear()
    failure_view = analysis.calibration_failures
    failure_view["s"] = "mutated view"
    assert analysis.procedure_results["s"].statistic == 0.7
    assert analysis.calibration_failures == {}


def test_catalog_statistics_view_prefers_procedure_results_over_raw_values():
    snapshot = (1.0, 2.0, 3.0)
    analysis = AnalysisResult(
        {},
        {},
        {"Normal": {"ks": 999.0, "extra": 0.3}, "Empty": {}},
        {},
        {"normal__ks": ProcedureResult(0.7)},
        snapshot,
        catalog_ids={"Normal": {"ks": "normal__ks"}},
    )
    assert analysis.sample_snapshot == snapshot
    assert analysis.statistics == {"Normal": {"extra": 0.3, "ks": 0.7}, "Empty": {}}
    view = analysis.statistics
    view["Normal"]["ks"] = -1
    view["Empty"]["invented"] = 42
    assert analysis.procedure_results["normal__ks"].statistic == 0.7
    assert analysis.statistics["Empty"] == {}
    analysis.procedure_results["normal__ks"] = ProcedureResult(0.8)
    assert analysis.statistics["Normal"]["ks"] == 0.8
    analysis.procedure_results.clear()
    assert "ks" not in analysis.statistics["Normal"]


def test_catalog_constructor_rejects_ambiguous_id_aliases():
    with pytest.raises(ValueError, match="Catalog procedure IDs must be unique"):
        AnalysisResult({}, {}, {}, {}, catalog_ids={"A": {"x": "same"}, "B": {"y": "same"}})
