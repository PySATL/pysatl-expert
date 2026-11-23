"""Contracts for additional statistics and ready recommendation methods."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from pysatl_expert.models.analysis_result import AnalysisResult, ProcedureResult


@dataclass(frozen=True)
class ProcedureRequest:
    """Optional calibration outputs required by a method in addition to raw T."""

    p_value: bool = False
    critical_values: bool = False
    alpha: float = 0.05
    required: bool = True

    def __post_init__(self):
        if any(
            not isinstance(v, bool) for v in (self.p_value, self.critical_values, self.required)
        ):
            raise ValueError("Procedure request flags must be booleans")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1")


class ProcedureAdapter(Protocol):
    """Library-independent procedure returning facts in Expert's format."""

    def evaluate(
        self, sample: np.ndarray, request: ProcedureRequest, *, allow_calibration: bool
    ) -> ProcedureResult:
        """Calculate requested facts or report unavailable calibration."""


class StatisticProcedure(Protocol):
    """Calculate one raw statistic; any required fitting belongs to the procedure."""

    def calculate(self, sample: np.ndarray) -> float:
        """Return a finite statistic for one sample."""


class RecommendationMethod(Protocol):
    """A ready method whose declared raw statistics must all be available.

    It may additionally expose ``procedure_requests``, a mapping of
    statistic IDs to ProcedureRequest objects.
    """

    required_statistics: Sequence[str]

    def predict(self, sample: np.ndarray, analysis: AnalysisResult) -> Any:
        """Use the sample and analysis to make a prediction."""


class AnalysisProvider(Protocol):
    """Produce reusable analysis evidence for an existing recommendation flow."""

    def __call__(self, sample: np.ndarray, /) -> AnalysisResult:
        """Analyze one validated sample and return its full analysis record."""
