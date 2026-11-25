"""Construct and execute one pysatl-criterion statistic for a distribution."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.goodness_of_fit_statistic import (
    AbstractGoodnessOfFitStatistic,
)

from pysatl_expert.core.criterion import AbstractCriterion
from pysatl_expert.criteria.adapters.criterion_adapter import CriterionAdapter
from pysatl_expert.criteria.parameters import (
    family_parameters_for_statistic,
    fitted_parameters_for_statistic,
)
from pysatl_expert.models.analysis_result import ProcedureResult


class GenericCriterion(AbstractCriterion):
    """Create a fresh Criterion engine from its class for every calculation."""

    def __init__(
        self,
        statistic_class: type[AbstractGoodnessOfFitStatistic],
        display_name: str | None = None,
    ):
        if not isinstance(statistic_class, type) or not issubclass(
            statistic_class, AbstractGoodnessOfFitStatistic
        ):
            raise TypeError("GenericCriterion requires a Criterion statistic class")
        name = display_name or statistic_class.short_code()
        super().__init__(name=name)
        self.statistic_class = statistic_class

    @property
    def requires_fit(self) -> bool:
        """Whether this statistic lacks a supported empty family hypothesis."""
        return family_parameters_for_statistic(self.statistic_class) is None

    def calculate(self, data, dist, params) -> float:
        """Return the scalar compatibility view of the structured calculation."""
        return self.evaluate(data, dist, params).statistic

    def evaluate(self, data, dist, params) -> ProcedureResult:
        """Return a native family statistic or a fitted raw statistic with identity.

        Family statistics use their explicitly supported empty hypothesis and
        the original observations. If the selected statistic only supports
        fixed parameters, Expert's fitted canonical coordinates are supplied
        for a raw statistic; the Criterion adapter marks the execution as
        ``fitted`` and refuses simple-null calibration for that value.
        """
        family = dist.name.lower().replace("_", "")
        parameter_values = family_parameters_for_statistic(self.statistic_class)
        fitted = parameter_values is None
        if parameter_values is None:
            observations, prepared = dist.prepare_criterion_input(data, params)
            parameter_values = fitted_parameters_for_statistic(
                self.statistic_class, family, prepared
            )
        else:
            observations = np.asarray(data, dtype=float)

        engine = self.statistic_class(parameter_values)
        adapter = CriterionAdapter(engine, fitted=fitted)
        result = adapter.execute_prepared(observations)
        if fitted:
            result = replace(
                result,
                preprocessing={
                    "method": (
                        f"{type(dist).__module__}.{type(dist).__qualname__}"
                        ".prepare_criterion_input"
                    ),
                    "parameters": deepcopy(params),
                },
            )
        return result
