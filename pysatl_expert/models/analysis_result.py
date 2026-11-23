from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from typing import Any


@dataclass(init=False)
class AnalysisResult:
    """Facts from one sample; procedure evidence has one canonical record per ID."""

    sample_stats: dict[str, float | int]
    parameters: dict[str, Any]
    _uncataloged_statistics: dict[str, dict[str, float]]
    failures: dict[str, str]
    procedure_results: dict[str, "ProcedureResult"] = field(default_factory=dict)
    sample_snapshot: tuple[float, ...] | None = None
    catalog_ids: dict[str, dict[str, str]] = field(default_factory=dict)

    def __init__(
        self,
        sample_stats: dict[str, float | int],
        parameters: dict[str, Any],
        statistics: dict[str, dict[str, float]],
        failures: dict[str, str],
        procedure_results: dict[str, "ProcedureResult"] | None = None,
        sample_snapshot: tuple[float, ...] | None = None,
        *,
        catalog_ids: dict[str, dict[str, str]] | None = None,
    ):
        self.sample_stats = sample_stats
        self.parameters = parameters
        self.failures = failures
        self.procedure_results = procedure_results if procedure_results is not None else {}
        self.sample_snapshot = sample_snapshot
        self.catalog_ids = {family: dict(ids) for family, ids in (catalog_ids or {}).items()}
        ids = [name for family in self.catalog_ids.values() for name in family.values()]
        if len(ids) != len(set(ids)):
            raise ValueError("Catalog procedure IDs must be unique")
        self._uncataloged_statistics = {
            family: {
                test: value
                for test, value in scores.items()
                if test not in self.catalog_ids.get(family, {})
            }
            for family, scores in statistics.items()
        }

    @property
    def statistics(self) -> dict[str, dict[str, float]]:
        """Return a detached catalog view; canonical values live in procedure_results."""
        values = {family: dict(scores) for family, scores in self._uncataloged_statistics.items()}
        for family, ids in self.catalog_ids.items():
            scores = values.setdefault(family, {})
            scores.update(
                (test, self.procedure_results[name].statistic)
                for test, name in ids.items()
                if name in self.procedure_results
            )
        return values

    @property
    def special_statistics(self) -> dict[str, float]:
        """Return a detached mapping from procedure ID to its raw statistic."""
        return {name: result.statistic for name, result in self.procedure_results.items()}

    @property
    def calibration_failures(self) -> dict[str, str]:
        """Return a detached mapping from procedure ID to its calibration failure reasons."""
        failures = {}
        for name, result in self.procedure_results.items():
            reasons = list(dict.fromkeys(result.unavailable.values()))
            if result.calibration_error:
                reasons.insert(0, result.calibration_error)
            if reasons:
                failures[name] = "; ".join(dict.fromkeys(reasons))
        return failures


@dataclass(frozen=True)
class ProcedureResult:
    """Statistic and optionally both calibration outputs from one procedure call."""

    statistic: float
    alternative: str | None = None
    p_value: float | None = None
    critical_values: tuple[float, ...] | None = None
    alpha: float | None = None
    source: str = ""
    hypothesis: tuple[tuple[str, float], ...] = ()
    hypothesis_description: str | None = None
    parameterization: str | None = None
    configuration: dict[str, Any] = field(default_factory=dict)
    parameter_mode: str | None = None
    calibration_method: str | None = None
    calibration_error: str | None = None
    unavailable: dict[str, str] = field(default_factory=dict)
    preprocessing: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.statistic, bool) or not isinstance(self.statistic, Real):
            raise TypeError("Statistic must be a finite real scalar")
        if not isfinite(self.statistic):
            raise ValueError("Statistic must be finite")
        if set(self.unavailable) - {"p_value", "critical_values"}:
            raise ValueError("Unknown unavailable result field")
        for name, reason in self.unavailable.items():
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("Unavailable result requires a reason")
            if getattr(self, name) is not None:
                raise ValueError("A result cannot be both available and unavailable")
        if self.alternative is not None and self.alternative not in {
            "less",
            "greater",
            "two-sided",
        }:
            raise ValueError("Procedure requires a valid alternative")
        if self.parameter_mode not in {None, "family", "known", "fitted"}:
            raise ValueError("Procedure parameter_mode must be family, known, fitted, or None")
        if self.p_value is not None:
            CalibrationResult(
                method=self.calibration_method or "",
                p_value=self.p_value,
                alternative=self.alternative,
            )
        if self.critical_values is not None:
            CalibrationResult(
                method=self.calibration_method or "",
                critical_values=self.critical_values,
                alpha=self.alpha,
                alternative=self.alternative,
            )


@dataclass(frozen=True)
class CalibrationResult:
    """One supplied calibration result and its method, without a validity claim."""

    method: str
    p_value: float | None = None
    critical_values: tuple[float, ...] | None = None
    alpha: float | None = None
    alternative: str | None = None

    def __post_init__(self):
        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError("Calibration method must be specified")
        if (self.p_value is None) == (self.critical_values is None):
            raise ValueError("Provide exactly one of p_value and critical_values")
        if self.p_value is not None and (not isfinite(self.p_value) or not 0 <= self.p_value <= 1):
            raise ValueError("p-value must be finite and between 0 and 1")
        if self.alternative not in {"less", "greater", "two-sided"}:
            raise ValueError("Calibration requires a valid alternative")
        if self.p_value is not None and self.alpha is not None:
            raise ValueError("alpha applies only to critical values")
        if self.critical_values is not None:
            values = self.critical_values
            if not isinstance(values, tuple) or len(values) not in (1, 2):
                raise ValueError("Critical values must be a tuple of one or two bounds")
            if not all(isfinite(value) for value in values):
                raise ValueError("Critical values must be finite")
            if len(values) == 2 and values[0] >= values[1]:
                raise ValueError("Critical bounds must be strictly increasing")
            if self.alpha is None or not 0 < self.alpha < 1:
                raise ValueError("Critical values require 0 < alpha < 1")
            expected = {1: {"less", "greater"}, 2: {"two-sided"}}
            if self.alternative not in expected[len(values)]:
                raise ValueError("Critical values require a matching alternative")


@dataclass
class RecommendationResult:
    """Retain analysis even if a recommendation cannot be produced."""

    analysis: AnalysisResult
    prediction: Any = None
    error: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
