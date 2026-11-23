"""Execute requested procedures and normalize their canonical result facts."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from numbers import Real
from typing import Protocol, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from pysatl_expert.core.recommendation import (
    ProcedureAdapter,
    ProcedureRequest,
    StatisticProcedure,
)
from pysatl_expert.models.analysis_result import AnalysisResult, ProcedureResult


Sample: TypeAlias = NDArray[np.float64]
ProcedureRegistry = Mapping[str, StatisticProcedure | ProcedureAdapter]


@dataclass(frozen=True)
class AnalysisPlan:
    """Materialized requests and procedures validated for their selected operation."""

    required_statistics: tuple[str, ...]
    procedure_requests: Mapping[str, ProcedureRequest]
    raw_procedures: Mapping[str, StatisticProcedure]
    adapted_procedures: Mapping[str, ProcedureAdapter]


def finite_statistic(value: object) -> float:
    """Return a finite real scalar or reject an unavailable raw result."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("statistic must be a finite numeric scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("statistic is not finite")
    return result


class _IdentityMatcher(Protocol):
    def __call__(self, previous: ProcedureResult) -> bool: ...


def _run_procedures(
    sample: Sample,
    required: tuple[str, ...],
    plan: AnalysisPlan,
    analysis: AnalysisResult,
    enabled: bool,
) -> None:
    """Execute each requested procedure once and store canonical result facts."""
    procedure_requests = plan.procedure_requests
    for name in required:
        try:
            if name in procedure_requests:
                request = procedure_requests[name]
                result = plan.adapted_procedures[name].evaluate(
                    sample.copy(), request, allow_calibration=enabled
                )
                if not isinstance(result, ProcedureResult):
                    raise TypeError("procedure must return ProcedureResult")
                if not enabled and (
                    result.p_value is not None or result.critical_values is not None
                ):
                    raise ValueError("procedure returned calibration without permission")
                if request.critical_values and result.critical_values is not None:
                    if result.alpha != request.alpha:
                        result = replace(
                            result,
                            critical_values=None,
                            alpha=None,
                            unavailable={
                                **result.unavailable,
                                "critical_values": (
                                    "procedure returned a different significance level"
                                ),
                            },
                        )
            else:
                statistic = finite_statistic(plan.raw_procedures[name].calculate(sample.copy()))
                result = ProcedureResult(statistic=statistic)
            analysis.procedure_results[name] = deepcopy(result)
        except Exception as exc:
            analysis.failures[name] = f"{type(exc).__name__}: {exc}"


def _procedure_needs_execution(
    name: str, request: ProcedureRequest, analysis: AnalysisResult
) -> bool:
    if name in analysis.failures:
        return False
    result = analysis.procedure_results.get(name)
    if result is None:
        return True
    if (
        request.critical_values
        and result.critical_values is not None
        and result.alpha != request.alpha
    ):
        return True
    return any(
        getattr(request, field)
        and getattr(result, field) is None
        and field not in result.unavailable
        for field in ("p_value", "critical_values")
    )


def _record_missing_procedure_facts(
    procedure_requests: Mapping[str, ProcedureRequest],
    analysis: AnalysisResult,
) -> None:
    for name, request in procedure_requests.items():
        result = analysis.procedure_results.get(name)
        if result is None:
            continue
        fields = tuple(field for field in ("p_value", "critical_values") if getattr(request, field))
        missing_facts = [field for field in fields if getattr(result, field) is None]
        if not missing_facts:
            continue
        unavailable = dict(result.unavailable)
        for field in missing_facts:
            unavailable.setdefault(
                field, result.calibration_error or "procedure did not provide requested result"
            )
        analysis.procedure_results[name] = replace(result, unavailable=unavailable)


def _calibration_fact_missing(
    name: str, request: ProcedureRequest, analysis: AnalysisResult
) -> bool:
    result = analysis.procedure_results.get(name)
    fields = tuple(field for field in ("p_value", "critical_values") if getattr(request, field))
    return bool(fields) and (
        result is None or any(getattr(result, field) is None for field in fields)
    )


def execute_procedures(
    sample: Sample,
    plan: AnalysisPlan,
    allow_calibration: bool,
    analysis: AnalysisResult,
) -> tuple[AnalysisResult, str | None]:
    """Execute a validated requirement plan and return its analysis and status."""
    required = plan.required_statistics
    procedure_requests = plan.procedure_requests
    for name in required:
        previous = analysis.procedure_results.get(name)
        procedure = (
            plan.adapted_procedures[name]
            if name in procedure_requests
            else plan.raw_procedures[name]
        )
        matcher: object = getattr(procedure, "matches_identity", None)
        if (
            previous is not None
            and callable(matcher)
            and not cast(_IdentityMatcher, matcher)(previous)
        ):
            analysis.procedure_results.pop(name)
    pending = [
        name
        for name in required
        if (
            name in procedure_requests
            and _procedure_needs_execution(name, procedure_requests[name], analysis)
        )
        or (
            name not in procedure_requests
            and name not in analysis.procedure_results
            and name not in analysis.failures
        )
    ]
    _run_procedures(
        sample,
        tuple(pending),
        plan,
        analysis,
        allow_calibration,
    )
    _record_missing_procedure_facts(procedure_requests, analysis)
    if any(name in analysis.failures for name in required):
        return analysis, "Required statistics unavailable"
    if any(
        request.required and _calibration_fact_missing(name, request, analysis)
        for name, request in procedure_requests.items()
    ):
        return analysis, "Required calibration unavailable"
    return analysis, None
