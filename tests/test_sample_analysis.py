from unittest.mock import MagicMock

import numpy as np
import pytest

from pysatl_expert._procedure_execution import AnalysisPlan, execute_procedures
from pysatl_expert.app import analyze_sample
from pysatl_expert.core.pipeline_components import PipelineComponents
from pysatl_expert.criteria.selectors.selector import CriterionSelector
from pysatl_expert.distributions.beta import BetaDistribution
from pysatl_expert.distributions.normal import NormalDistribution
from pysatl_expert.models.analysis_result import AnalysisResult
from pysatl_expert.models.feature_extractor import FeatureExtractor
from pysatl_expert.pipeline import DistributionPipeline
from pysatl_expert.reporting.result_report import render_result_report


SAMPLE = [0.1, 0.4, 0.8, 1.2, 1.5]


def test_real_analysis_without_model(monkeypatch, capsys):
    from pysatl_expert import app

    monkeypatch.setattr(app, "MLStrategy", lambda **kwargs: pytest.fail("model was loaded"))
    result = analyze_sample(SAMPLE, distributions=["Normal"], statistics=["normal__ks"])
    assert result.sample_stats["sample_size"] == 5
    assert result.parameters["Normal"]["mu"] == pytest.approx(0.8)
    assert np.isfinite(result.statistics["Normal"]["ks"])
    assert result.failures == {}
    assert capsys.readouterr().out == ""


def test_real_short_normal_sample_reports_unavailable_chen_shapiro():
    result = analyze_sample(
        [1, 2, 3], distributions=["Normal"], statistics=["normal__cs", "normal__ks"]
    )

    assert "Normal" in result.parameters
    assert np.isfinite(result.statistics["Normal"]["ks"])
    assert "cs" not in result.statistics["Normal"]
    assert "normal__cs" in result.failures
    assert "at least 4 observations" in result.failures["normal__cs"]


@pytest.mark.parametrize(
    ("distributions", "statistics"),
    [
        ([], ["normal__ks"]),
        (["Unknown"], ["normal__ks"]),
        (["Normal"], ["unknown"]),
        (["Beta"], ["normal__ks"]),
        (["Normal"], []),
        ("Normal", ["normal__ks"]),
    ],
)
def test_invalid_request_does_not_fit(monkeypatch, distributions, statistics):
    from pysatl_expert.distributions.normal import NormalDistribution

    monkeypatch.setattr(NormalDistribution, "fit", lambda *args: pytest.fail("fit called"))
    with pytest.raises(ValueError):
        analyze_sample(SAMPLE, distributions=distributions, statistics=statistics)


def test_unrequested_distribution_and_statistic_are_not_executed(monkeypatch):
    from pysatl_expert.criteria.calculate.generic import GenericCriterion
    from pysatl_expert.distributions.beta import BetaDistribution

    calls = []
    original = GenericCriterion.evaluate

    def record(self, *args):
        calls.append(self.name)
        return original(self, *args)

    monkeypatch.setattr(BetaDistribution, "fit", lambda *args: pytest.fail("Beta fit called"))
    monkeypatch.setattr(GenericCriterion, "evaluate", record)
    analyze_sample(SAMPLE, distributions=["Normal"], statistics=["normal__ks"])
    assert calls == ["ks"]


def test_public_analysis_reports_fit_failure_and_keeps_other_family(monkeypatch):
    from pysatl_expert.distributions.beta import BetaDistribution

    def fail_fit(*args):
        raise ValueError("fit failed deliberately")

    monkeypatch.setattr(BetaDistribution, "fit", fail_fit)
    result = analyze_sample(
        [0.1, 0.2, 0.4, 0.6, 0.8],
        distributions=["Normal", "Beta"],
        statistics=["normal__ks", "beta__ks"],
    )
    assert np.isfinite(result.statistics["Normal"]["ks"])
    assert "fit failed deliberately" in result.failures["fit:Beta"]
    assert result.failures["beta__ks"] == "not run: fit failed"


@pytest.mark.parametrize("requested_checks", [False, True], ids=["catalog", "explicit-checks"])
def test_public_analysis_preserves_selector_failure_and_success(monkeypatch, requested_checks):
    original = CriterionSelector.get_applicable_criteria

    def select(self, data, distribution):
        if distribution.name == "Beta":
            raise RuntimeError("selector failed deliberately")
        return original(self, data, distribution)

    monkeypatch.setattr(CriterionSelector, "get_applicable_criteria", select)
    sample = [0.1, 0.2, 0.4, 0.6, 0.8]
    checks = ["normal__ks", "beta__ks"]
    if requested_checks:
        result = analyze_sample(sample, distributions=["Normal", "Beta"], statistics=checks)
    else:
        components = PipelineComponents(
            [NormalDistribution(), BetaDistribution()],
            CriterionSelector(feature_names=checks),
            None,
            FeatureExtractor(),
        )
        result = DistributionPipeline(components).analyze_catalog(sample)

    reason = "RuntimeError: selector failed deliberately"
    assert set(result.parameters) == {"Normal", "Beta"}
    assert np.isfinite(result.parameters["Normal"]["mu"])
    assert all(np.isfinite(value) for value in result.parameters["Beta"].values())
    assert np.isfinite(result.statistics["Normal"]["ks"])
    assert result.statistics["Beta"] == {}
    expected_failures = {"selector:Beta": reason}
    if requested_checks:
        expected_failures["beta__ks"] = f"not run: selector failed: {reason}"
    assert result.failures == expected_failures
    report = render_result_report(result)
    assert f"selector:Beta: {reason}" in report
    assert "no recorded errors" not in report


def test_partial_failures_preserve_success():
    good = MagicMock(name="good")
    good.name, good.support = "Good", (-np.inf, np.inf)
    good.fit.return_value = {"mu": 1}
    bad = MagicMock(name="bad")
    bad.name, bad.support = "Bad", (-np.inf, np.inf)
    bad.fit.side_effect = ValueError("bad fit")
    out = MagicMock(name="out")
    out.name, out.support = "Out", (10, 20)
    ok = MagicMock(name="ok")
    ok.name, ok.calculate.return_value = "ok", 0.5
    failed = MagicMock(name="failed")
    failed.name = "fail"
    failed.calculate.side_effect = ValueError("bad statistic")
    selector = MagicMock()
    selector.get_applicable_criteria.return_value = [ok, failed]
    components = PipelineComponents([good, bad, out], selector, None, FeatureExtractor())
    checks = {"Good": ["good__ok", "good__fail"], "Bad": ["bad__x"], "Out": ["out__x"]}
    result = DistributionPipeline(components)._analyze_sample(SAMPLE, checks)
    assert result.statistics["Good"] == {"ok": 0.5}
    assert result.parameters["Good"] == {"mu": 1}
    assert "bad statistic" in result.failures["good__fail"]
    assert "bad fit" in result.failures["fit:Bad"]
    assert result.failures["bad__x"] == "not run: fit failed"
    assert "support" in result.failures["out__x"]
    assert selector.get_applicable_criteria.call_count == 2


@pytest.mark.parametrize(
    "raw_value",
    [None, True, "0.5", np.nan, np.inf, np.array([0.5])],
    ids=["none", "bool", "string", "nan", "infinity", "array"],
)
def test_non_numeric_criterion_results_are_reported_without_losing_success(raw_value):
    good = MagicMock(name="good")
    good.name, good.support = "Good", (-np.inf, np.inf)
    good.fit.return_value = {"mu": 1}
    ok = MagicMock(name="ok")
    ok.name, ok.calculate.return_value = "ok", np.float64(0.25)
    bad = MagicMock(name="bad")
    bad.name, bad.calculate.return_value = "bad", raw_value
    selector = MagicMock()
    selector.get_applicable_criteria.return_value = [ok, bad]
    components = PipelineComponents([good], selector, None, FeatureExtractor())

    result = DistributionPipeline(components)._analyze_sample(
        SAMPLE, {"Good": ["good__ok", "good__bad"]}
    )

    assert result.statistics["Good"] == {"ok": 0.25}
    assert result.parameters["Good"] == {"mu": 1}
    assert "good__bad" in result.failures


@pytest.mark.parametrize("raw_value", [None, True, "0.5", np.nan, np.inf, np.array([0.5])])
def test_non_numeric_injected_procedure_results_are_rejected(raw_value):
    procedure = MagicMock()
    procedure.calculate.return_value = raw_value
    analysis = AnalysisResult({}, {}, {}, {})

    execute_procedures(
        np.array(SAMPLE),
        AnalysisPlan(("procedure_id",), {}, {"procedure_id": procedure}, {}),
        False,
        analysis,
    )

    assert "procedure_id" not in analysis.special_statistics
    assert "procedure_id" in analysis.failures


@pytest.mark.parametrize("data", [[], [[1, 2]], [1, np.inf], [2, 2]])
def test_invalid_sample_rejected(data):
    with pytest.raises(ValueError):
        analyze_sample(data, distributions=["Normal"], statistics=["normal__ks"])


def test_missing_catalog_selector_reports_configuration_without_losing_fit():
    distribution = MagicMock()
    distribution.name, distribution.support = "Good", (-np.inf, np.inf)
    distribution.fit.return_value = {"mu": 1.0}
    pipeline = DistributionPipeline(
        PipelineComponents([distribution], None, None, FeatureExtractor())
    )

    result = pipeline._analyze_sample(SAMPLE, {"Good": ["good__check"]})

    assert result.parameters["Good"] == {"mu": 1.0}
    assert result.statistics["Good"] == {}
    assert result.failures["good__check"] == (
        "not run: selector failed: ValueError: Catalog analysis requires a criterion selector"
    )
    descriptive = pipeline._analyze_sample(SAMPLE, distributions=[])
    assert descriptive.sample_stats["sample_size"] == len(SAMPLE)
    assert descriptive.failures == {}


def test_failed_fit_keeps_family_fact_and_blocks_fitted_statistic(monkeypatch):
    def fail_fit(*args):
        raise ValueError("fit failed deliberately")

    monkeypatch.setattr(NormalDistribution, "fit", fail_fit)
    result = analyze_sample(
        SAMPLE, distributions=["Normal"], statistics=["normal__ad", "normal__ks"]
    )

    assert result.parameters == {}
    assert result.failures == {
        "fit:Normal": "ValueError: fit failed deliberately",
        "normal__ks": "not run: fit failed",
    }
    fact = result.procedure_results["normal__ad"]
    assert np.isfinite(fact.statistic)
    assert fact.parameter_mode == "family"
    assert result.statistics == {"Normal": {"ad": fact.statistic}}
    assert result.catalog_ids == {"Normal": {"ad": "normal__ad"}}


@pytest.mark.parametrize("family_opt_in", [False, True])
def test_failed_fit_runs_scalar_criterion_only_with_explicit_opt_in(family_opt_in):
    distribution = MagicMock()
    distribution.name, distribution.support = "Legacy", (-np.inf, np.inf)
    distribution.fit.side_effect = ValueError("bad fit")
    criterion = MagicMock()
    criterion.name = "score"
    criterion.calculate.return_value = 0.4
    if family_opt_in:
        criterion.requires_fit = False
    selector = MagicMock()
    selector.get_applicable_criteria.return_value = [criterion]
    result = DistributionPipeline(
        PipelineComponents([distribution], selector, None, FeatureExtractor())
    ).analyze_catalog(SAMPLE)

    selector.get_applicable_criteria.assert_called_once()
    assert criterion.calculate.call_count == int(family_opt_in)
    if family_opt_in:
        assert result.statistics == {"Legacy": {"score": 0.4}}
        assert "legacy__score" not in result.failures
    else:
        assert result.failures["legacy__score"] == "not run: fit failed"
        assert result.statistics == {"Legacy": {}}
    assert result.procedure_results == {}


def test_support_guard_still_blocks_family_statistic(monkeypatch):
    from pysatl_expert.distributions.weibull import WeibullDistribution

    selector = MagicMock()
    monkeypatch.setattr(WeibullDistribution, "fit", lambda *args: pytest.fail("fit called"))
    result = DistributionPipeline(
        PipelineComponents([WeibullDistribution()], selector, None, FeatureExtractor())
    ).analyze_catalog([-0.5, 0.1, 0.7])

    selector.get_applicable_criteria.assert_not_called()
    assert "sample outside distribution support" in result.failures["fit:Weibull"]
    assert result.statistics == {"Weibull": {}}
