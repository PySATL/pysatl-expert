from unittest.mock import Mock

import numpy as np
import pytest

from pysatl_expert.core.recommendation import ProcedureRequest
from pysatl_expert.models.analysis_result import AnalysisResult, ProcedureResult
from pysatl_expert.pipeline import DistributionPipeline


def make_pipeline():
    components = Mock()
    components.feature_extractor.calculate_sample_stats.return_value = {
        "min": 1.0,
        "max": 4.0,
        "mean": 2.5,
    }
    return DistributionPipeline(components), components


def test_analyze_procedures_returns_analysis_without_prediction():
    pipeline, _ = make_pipeline()
    procedure = Mock()
    procedure.calculate.return_value = 7.5

    result = pipeline.analyze_procedures(
        [1, 2, 3, 4], required_statistics=["custom"], procedures={"custom": procedure}
    )

    assert isinstance(result, AnalysisResult)
    assert result.special_statistics == {"custom": 7.5}
    procedure.calculate.assert_called_once()


def test_procedure_result_rejects_boolean_as_statistic():
    with pytest.raises(TypeError, match="finite real scalar"):
        ProcedureResult(statistic=True)


def test_procedure_request_records_optional_unavailable_p_value():
    pipeline, _ = make_pipeline()
    procedure = Mock()
    procedure.evaluate.return_value = ProcedureResult(
        statistic=3.0,
        alternative="greater",
        calibration_error="p-value unavailable",
    )

    result = pipeline.analyze_procedures(
        [1, 2, 3, 4],
        required_statistics=["custom"],
        procedures={"custom": procedure},
        procedure_requests={"custom": ProcedureRequest(p_value=True, required=False)},
    )

    assert result.special_statistics["custom"] == 3.0
    assert result.calibration_failures["custom"] == "p-value unavailable"


def test_invalid_configuration_precedes_procedure_calls():
    pipeline, _ = make_pipeline()
    procedure = Mock()

    with pytest.raises(ValueError, match="Unavailable statistics"):
        pipeline.analyze_procedures(
            [1, 2, 3, 4],
            required_statistics=["missing"],
            procedures={"custom": procedure},
        )

    procedure.calculate.assert_not_called()


def test_invalid_sample_precedes_procedure_execution():
    pipeline, _ = make_pipeline()
    procedure = Mock()

    with pytest.raises(ValueError, match="non-constant"):
        pipeline.analyze_procedures(
            np.array([2.0, 2.0]),
            required_statistics=["custom"],
            procedures={"custom": procedure},
            procedure_requests={"custom": ProcedureRequest(p_value=True)},
            allow_calibration=True,
        )

    procedure.evaluate.assert_not_called()
