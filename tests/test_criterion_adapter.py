from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from pysatl_criterion.statistics.alternative import RightAlternative

from pysatl_expert.core.pipeline_components import PipelineComponents
from pysatl_expert.core.recommendation import ProcedureRequest
from pysatl_expert.criteria.criterion_adapter import CriterionAdapter
from pysatl_expert.models.analysis_result import AnalysisResult, ProcedureResult
from pysatl_expert.models.feature_extractor import FeatureExtractor
from pysatl_expert.pipeline import DistributionPipeline


def _empty_pipeline() -> DistributionPipeline:
    return DistributionPipeline(PipelineComponents([], None, None, FeatureExtractor()))


def test_catalog_preprocessing_cannot_replace_a_direct_criterion_call():
    from pysatl_criterion.distribution import NormalDistributionDescriptor

    from pysatl_expert.app import analyze_sample
    from pysatl_expert.criteria.catalog import CRITERIA_SPEC_BY_KEY

    sample = np.array([10.0, 11.0, 12.0, 14.0, 20.0])
    seed = analyze_sample(sample, distributions=["Normal"], statistics=["normal__ks"])
    fact = seed.procedure_results["normal__ks"]
    statistic_class = CRITERIA_SPEC_BY_KEY[("normal", "ks")].statistic_class
    parameters = NormalDistributionDescriptor.default_parameterization().parse(
        {"mean": 0.0, "var": 1.0}
    )
    adapter = CriterionAdapter(statistic_class(parameters), fitted=True)
    method = SimpleNamespace(
        required_statistics=("normal__ks",),
        procedure_requests={"normal__ks": ProcedureRequest()},
        predict=lambda sample, analysis: analysis.procedure_results["normal__ks"].statistic,
    )
    pipeline = _empty_pipeline()
    fresh = pipeline.recommend(sample, method=method, procedures={"normal__ks": adapter})
    tracked = MagicMock(wraps=adapter)
    reused = pipeline.recommend(
        sample,
        method=method,
        procedures={"normal__ks": tracked},
        analysis_provider=lambda sample: seed,
    )

    assert fact.statistic == pytest.approx(0.2531285467567539)
    assert fact.preprocessing["parameters"] == seed.parameters["Normal"]
    assert not adapter.matches_identity(fact)
    assert fresh.error is None and reused.error is None
    assert fresh.prediction == reused.prediction == 1.0
    tracked.evaluate.assert_called_once()
    assert seed.procedure_results["normal__ks"] is fact

    tracked.reset_mock()
    compatible = pipeline.recommend(
        sample,
        method=method,
        procedures={"normal__ks": tracked},
        analysis_provider=lambda sample: fresh.analysis,
    )
    assert compatible.error is None and compatible.prediction == fresh.prediction
    tracked.evaluate.assert_not_called()


def statistic():
    class EmptyHypothesis:
        parameter_values = None

        @staticmethod
        def parameters():
            return {}

    class Statistic:
        @staticmethod
        def alternative():
            return RightAlternative()

        @staticmethod
        def hypothesis():
            return EmptyHypothesis()

        @staticmethod
        def code():
            return "demo"

        @staticmethod
        def configuration():
            return {}

        @staticmethod
        def execute_statistic(rvs):
            return 0.4

    return Statistic()


def test_one_calibration_returns_statistic_p_and_cv():
    stat = statistic()
    calibrate = MagicMock(
        return_value=SimpleNamespace(
            statistic=0.4,
            p_value=0.12,
            critical_value=0.9,
            significance_level=0.05,
        )
    )
    adapter = CriterionAdapter(
        stat,
        calibrate=calibrate,
        calibration_method="test stub",
    )
    result = adapter.evaluate(
        [1, 2, 3],
        ProcedureRequest(p_value=True, critical_values=True),
        allow_calibration=True,
    )
    assert result == ProcedureResult(
        statistic=0.4,
        alternative="greater",
        p_value=0.12,
        critical_values=(0.9,),
        alpha=0.05,
        source="demo",
        configuration={},
        parameter_mode="family",
        calibration_method="test stub",
    )
    calibrate.assert_called_once()


@pytest.mark.parametrize(
    ("procedure_request", "expected_unavailable"),
    [
        (ProcedureRequest(p_value=True), {}),
        (
            ProcedureRequest(p_value=True, critical_values=True),
            {"critical_values": "calibration returned a different significance level"},
        ),
    ],
)
def test_wrong_alpha_bounds_do_not_discard_calibrated_p_value(
    procedure_request, expected_unavailable
):
    calibrate = MagicMock(
        return_value=SimpleNamespace(
            statistic=0.4,
            p_value=0.12,
            critical_value=0.9,
            significance_level=0.1,
        )
    )
    result = CriterionAdapter(
        statistic(), calibrate=calibrate, calibration_method="test stub"
    ).evaluate([1, 2, 3], procedure_request, allow_calibration=True)

    assert result.statistic == 0.4
    assert result.p_value == 0.12
    assert result.critical_values is None
    assert result.alpha is None
    assert result.unavailable == expected_unavailable


@pytest.mark.parametrize("invalid_statistic", [True, "0.4", 0.4 + 0j])
def test_invalid_calibrator_statistic_falls_back_to_valid_raw_statistic(invalid_statistic):
    calibrate = MagicMock(
        return_value=SimpleNamespace(
            statistic=invalid_statistic,
            p_value=0.12,
            critical_value=None,
            significance_level=0.05,
        )
    )
    result = CriterionAdapter(
        statistic(), calibrate=calibrate, calibration_method="test stub"
    ).evaluate([1, 2, 3], ProcedureRequest(p_value=True), allow_calibration=True)

    assert result.statistic == 0.4
    assert result.p_value is None
    assert result.unavailable == {"p_value": result.calibration_error}
    assert result.calibration_error is not None


def test_seeded_calibration_from_another_method_is_recomputed():
    calibrate = MagicMock(
        return_value=SimpleNamespace(
            statistic=0.4,
            p_value=0.12,
            critical_value=None,
            significance_level=0.05,
        )
    )
    adapter = CriterionAdapter(
        statistic(), calibrate=calibrate, calibration_method="active-calibrator"
    )
    method = SimpleNamespace(
        required_statistics=("demo",),
        procedure_requests={"demo": ProcedureRequest(p_value=True)},
        predict=lambda sample, analysis: analysis.procedure_results["demo"].p_value,
    )
    seed = AnalysisResult(
        sample_stats={},
        parameters={},
        statistics={},
        failures={},
        sample_snapshot=(1.0, 2.0, 3.0),
        procedure_results={
            "demo": ProcedureResult(
                statistic=0.4,
                alternative="greater",
                p_value=0.3,
                source="demo",
                calibration_method="seed-calibrator",
            )
        },
    )

    result = _empty_pipeline().recommend(
        [1, 2, 3],
        method=method,
        procedures={"demo": adapter},
        allow_calibration=True,
        analysis_provider=lambda sample: seed,
    )

    assert result.error is None
    assert result.prediction == 0.12
    assert result.analysis.procedure_results["demo"].calibration_method == "active-calibrator"
    calibrate.assert_called_once()
    assert adapter.matches_identity(
        ProcedureResult(
            statistic=0.4,
            source="demo",
            alternative="greater",
            parameter_mode="family",
        )
    )


def test_disabled_calibration_keeps_statistic():
    stat = statistic()
    calibrate = MagicMock()
    result = CriterionAdapter(stat, calibrate=calibrate, calibration_method="stub").evaluate(
        [1, 2, 3],
        ProcedureRequest(p_value=True),
        allow_calibration=False,
    )
    assert result.statistic == 0.4
    assert result.calibration_error == "calibration not enabled"
    calibrate.assert_not_called()


def test_calibration_error_does_not_discard_statistic():
    stat = statistic()
    calibrate = MagicMock(side_effect=ValueError("unsupported"))
    result = CriterionAdapter(stat, calibrate=calibrate, calibration_method="stub").evaluate(
        [1, 2, 3],
        ProcedureRequest(p_value=True),
        allow_calibration=True,
    )
    assert result.statistic == 0.4
    assert "unsupported" in result.calibration_error


def test_other_adapter_and_required_calibration_flow():
    class OtherAdapter:
        def evaluate(self, sample, request, *, allow_calibration):
            return ProcedureResult(
                statistic=0.6,
                alternative="less",
                source="other",
                calibration_error="unavailable" if request.p_value else None,
            )

    method = SimpleNamespace(
        required_statistics=("other",),
        procedure_requests={"other": ProcedureRequest(p_value=True)},
        predict=MagicMock(),
    )
    result = _empty_pipeline().recommend(
        [1, 2, 3], method=method, procedures={"other": OtherAdapter()}
    )
    assert result.analysis.special_statistics == {"other": 0.6}
    assert result.analysis.calibration_failures == {"other": "unavailable"}
    assert result.error == "Required calibration unavailable"
    method.predict.assert_not_called()


def test_method_uses_calibrated_result_from_adapter():
    stat = statistic()
    calibrate = MagicMock(
        return_value=SimpleNamespace(
            statistic=0.4,
            p_value=0.12,
            critical_value=0.9,
            significance_level=0.05,
        )
    )
    method = SimpleNamespace(
        required_statistics=("demo",),
        procedure_requests={"demo": ProcedureRequest(p_value=True, critical_values=True)},
        predict=lambda sample, analysis: analysis.procedure_results["demo"].p_value,
    )
    result = _empty_pipeline().recommend(
        [1, 2, 3],
        method=method,
        procedures={"demo": CriterionAdapter(stat, calibrate=calibrate, calibration_method="stub")},
        allow_calibration=True,
    )
    assert result.error is None
    assert result.prediction == 0.12
    assert result.analysis.procedure_results["demo"].critical_values == (0.9,)


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("p_value", [None, 0.12])
def test_wrong_adapter_alpha_preserves_statistic_and_available_p_value(required, p_value):
    supplied = ProcedureResult(
        statistic=0.4,
        alternative="greater",
        p_value=p_value,
        critical_values=(0.9,),
        alpha=0.1,
        source="demo",
        calibration_method="stub",
    )
    adapter = SimpleNamespace(evaluate=MagicMock(return_value=supplied))
    method = SimpleNamespace(
        required_statistics=("demo",),
        procedure_requests={
            "demo": ProcedureRequest(
                p_value=p_value is not None,
                critical_values=True,
                alpha=0.05,
                required=required,
            )
        },
        predict=MagicMock(side_effect=lambda sample, analysis: analysis.procedure_results["demo"]),
    )
    result = _empty_pipeline().recommend(
        [1, 2, 3],
        method=method,
        procedures={"demo": adapter},
        allow_calibration=True,
    )
    assert result.analysis.failures == {}
    assert result.analysis.special_statistics == {"demo": 0.4}
    actual = result.analysis.procedure_results["demo"]
    assert actual.statistic == 0.4
    assert actual.p_value == p_value
    assert actual.critical_values is None
    assert actual.alpha is None
    assert "different significance level" in actual.unavailable["critical_values"]
    assert result.analysis.calibration_failures["demo"] == actual.unavailable["critical_values"]
    if required:
        assert result.error == "Required calibration unavailable"
        method.predict.assert_not_called()
    else:
        assert result.error is None
        assert result.prediction == actual
        method.predict.assert_called_once()
    assert supplied.critical_values == (0.9,)
    assert supplied.alpha == 0.1
    assert supplied.unavailable == {}
    adapter.evaluate.assert_called_once()


def test_unrequested_wrong_alpha_bounds_do_not_block_required_p_value():
    supplied = ProcedureResult(
        statistic=0.4,
        alternative="greater",
        p_value=0.12,
        critical_values=(0.9,),
        alpha=0.1,
        source="demo",
        calibration_method="stub",
    )
    method = SimpleNamespace(
        required_statistics=("demo",),
        procedure_requests={"demo": ProcedureRequest(p_value=True)},
        predict=lambda sample, analysis: analysis.procedure_results["demo"].p_value,
    )
    result = _empty_pipeline().recommend(
        [1, 2, 3],
        method=method,
        procedures={"demo": SimpleNamespace(evaluate=lambda *args, **kwargs: supplied)},
        allow_calibration=True,
    )
    assert result.error is None
    assert result.prediction == 0.12
    assert result.analysis.calibration_failures == {}


def test_real_criterion_normal_ks_exposes_hypothesis_configuration_and_raw_value():
    from pysatl_criterion.distribution import NormalDistributionDescriptor

    from pysatl_expert.criteria.calculate.generic import GenericCriterion
    from pysatl_expert.criteria.catalog import CRITERIA_SPEC_BY_KEY
    from pysatl_expert.criteria.parameters import fitted_parameters_for_statistic
    from pysatl_expert.distributions.normal import NormalDistribution

    engine_class = CRITERIA_SPEC_BY_KEY[("normal", "ks")].statistic_class
    known_parameters = NormalDistributionDescriptor.default_parameterization().parse(
        {"mean": 0.0, "var": 1.0}
    )
    known_engine = engine_class(known_parameters)
    known = CriterionAdapter(known_engine).execute_prepared(np.array([-1.2, -0.4, 0.1, 0.5, 1.1]))
    assert known.source == engine_class.code()
    assert known.parameter_mode == "known"
    assert known.hypothesis == (("mean", 0.0), ("var", 1.0))
    assert known.parameterization == known_parameters.parameterization.id
    assert known.configuration == engine_class(known_parameters).configuration()

    sample = np.array([-1.2, -0.4, 0.1, 0.5, 1.1])
    distribution = NormalDistribution()
    params = distribution.fit(sample)
    observations, engine_parameters = distribution.prepare_criterion_input(sample, params)
    fitted_parameters = fitted_parameters_for_statistic(engine_class, "normal", engine_parameters)
    direct_engine = engine_class(fitted_parameters)
    direct = CriterionAdapter(direct_engine, fitted=True).execute_prepared(observations)
    catalog_value = GenericCriterion(engine_class).calculate(sample, distribution, params)
    assert catalog_value == pytest.approx(direct.statistic)
    assert direct.parameter_mode == "fitted"
    assert direct.hypothesis == tuple(sorted(fitted_parameters.as_dict().items()))
    assert direct.configuration == direct_engine.configuration()


def test_fitted_criterion_cannot_use_calibrator_or_reuse_calibrated_seed():
    stat = statistic()
    calibrate = MagicMock()
    adapter = CriterionAdapter(stat, calibrate=calibrate, calibration_method="stub", fitted=True)
    result = adapter.evaluate([1, 2, 3], ProcedureRequest(p_value=True), allow_calibration=True)

    assert result.statistic == 0.4
    assert result.parameter_mode == "fitted"
    assert result.p_value is None
    assert result.unavailable == {
        "p_value": "fitted parameters do not have a compatible calibration"
    }
    calibrate.assert_not_called()
    calibrated_seed = ProcedureResult(
        statistic=0.4,
        source="demo",
        alternative="greater",
        p_value=0.2,
        parameter_mode="fitted",
        calibration_method="stub",
    )
    assert not adapter.matches_identity(calibrated_seed)


def test_required_fitted_calibration_failure_blocks_recommendation():
    calibrate = MagicMock()
    adapter = CriterionAdapter(
        statistic(), calibrate=calibrate, calibration_method="stub", fitted=True
    )
    method = SimpleNamespace(
        required_statistics=("demo",),
        procedure_requests={"demo": ProcedureRequest(p_value=True)},
        predict=MagicMock(),
    )

    result = _empty_pipeline().recommend(
        [1, 2, 3],
        method=method,
        procedures={"demo": adapter},
        allow_calibration=True,
    )

    assert result.analysis.procedure_results["demo"].parameter_mode == "fitted"
    assert result.error == "Required calibration unavailable"
    assert result.analysis.calibration_failures["demo"]
    method.predict.assert_not_called()
    calibrate.assert_not_called()
