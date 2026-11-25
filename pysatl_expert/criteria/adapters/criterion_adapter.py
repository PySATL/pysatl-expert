"""Adapt a configured pysatl-criterion statistic to the Expert procedure contract."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from typing import Any, Protocol

import numpy as np

from pysatl_expert._procedure_execution import finite_statistic
from pysatl_expert.core.recommendation import ProcedureRequest
from pysatl_expert.models.analysis_result import ProcedureResult


class CriterionStatistic(Protocol):
    """The immutable statistic API exposed by pysatl-criterion."""

    def alternative(self) -> Any: ...

    def hypothesis(self) -> Any: ...

    def code(self) -> str: ...

    def configuration(self) -> dict[str, Any]: ...

    def execute_statistic(self, rvs: np.ndarray) -> float: ...


class CriterionAdapter:
    """Execute one Criterion instance and calibrate only its declared hypothesis."""

    def __init__(
        self,
        statistic: CriterionStatistic,
        *,
        calibrate: Callable[..., Any] | None = None,
        calibration_method: str | None = None,
        fitted: bool = False,
    ):
        if calibrate is not None and not calibration_method:
            raise ValueError("An explicit calibration method name is required")
        if not isinstance(fitted, bool):
            raise TypeError("fitted must be a bool")
        self.statistic = statistic
        self.calibrate = calibrate
        self.calibration_method = calibration_method
        self.fitted = fitted

    def _metadata(self) -> dict[str, Any]:
        alternatives = {"left": "less", "right": "greater", "two_tailed": "two-sided"}
        alternative_type = self.statistic.alternative().type().value
        try:
            alternative = alternatives[alternative_type]
        except KeyError as exc:
            raise ValueError(f"Unsupported Criterion alternative: {alternative_type}") from exc

        hypothesis = self.statistic.hypothesis()
        values = hypothesis.parameter_values
        parameter_values = values.as_dict() if values is not None else hypothesis.parameters()
        parameterization = values.parameterization.id if values is not None else None
        mode = "fitted" if self.fitted else ("family" if not parameter_values else "known")
        return dict(
            alternative=alternative,
            source=self.statistic.code(),
            hypothesis=tuple(sorted(parameter_values.items())),
            hypothesis_description=None,
            parameterization=parameterization,
            configuration=deepcopy(self.statistic.configuration()),
            parameter_mode=mode,
            preprocessing={},
        )

    def matches_identity(self, result: ProcedureResult) -> bool:
        """Check seeded evidence against the full procedure identity."""
        metadata = self._metadata()
        if not all(getattr(result, name) == value for name, value in metadata.items()):
            return False
        has_calibration = result.p_value is not None or result.critical_values is not None
        if self.fitted and has_calibration:
            return False
        return not has_calibration or result.calibration_method == self.calibration_method

    def execute_prepared(self, observations: np.ndarray) -> ProcedureResult:
        """Execute Criterion on prepared observations without mutating its engine."""
        raw = self.statistic.execute_statistic(rvs=np.asarray(observations, dtype=float).copy())
        return ProcedureResult(statistic=finite_statistic(raw), **self._metadata())

    def evaluate(
        self, sample: np.ndarray, request: ProcedureRequest, *, allow_calibration: bool
    ) -> ProcedureResult:
        metadata = self._metadata()
        wants_calibration = request.p_value or request.critical_values
        if wants_calibration and self.fitted:
            reason = "fitted parameters do not have a compatible calibration"
        elif wants_calibration and allow_calibration and self.calibrate is not None:
            try:
                result = self.calibrate(
                    self.statistic, sample.copy(), significance_level=request.alpha
                )
                calibrated_alternative = getattr(result, "alternative", None)
                if (
                    calibrated_alternative is not None
                    and calibrated_alternative != metadata["alternative"]
                ):
                    raise ValueError("calibration returned a different alternative")
                bounds = result.critical_value
                if bounds is not None and not isinstance(bounds, tuple):
                    bounds = (float(bounds),)
                unavailable = {}
                if bounds is not None and result.significance_level != request.alpha:
                    bounds = None
                    if request.critical_values:
                        unavailable["critical_values"] = (
                            "calibration returned a different significance level"
                        )
                if request.p_value and result.p_value is None:
                    unavailable["p_value"] = "calibrator did not provide p-value"
                if request.critical_values and bounds is None:
                    unavailable.setdefault(
                        "critical_values", "calibrator did not provide critical values"
                    )
                return ProcedureResult(
                    statistic=result.statistic,
                    p_value=result.p_value,
                    critical_values=bounds,
                    alpha=request.alpha if bounds is not None else None,
                    calibration_method=self.calibration_method,
                    unavailable=unavailable,
                    **metadata,
                )
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
        elif wants_calibration:
            reason = (
                "calibration not enabled" if not allow_calibration else "calibrator unavailable"
            )
        else:
            reason = None

        unavailable = {
            key: reason
            for key in ("p_value", "critical_values")
            if getattr(request, key) and reason is not None
        }
        raw_result = self.execute_prepared(sample)
        return replace(raw_result, calibration_error=reason, unavailable=unavailable)
