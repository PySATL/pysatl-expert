import inspect
import logging

from pysatl_expert.core.criterion import AbstractCriterion


logger = logging.getLogger(__name__)


class GenericCriterion(AbstractCriterion):
    """Run a pysatl-criterion statistic through the expert-system interface.

    The distribution prepares matching observations and engine parameters.
    This adapter applies those parameters and supplies theoretical CDF values
    when required by the statistic method signature.
    """

    PARAM_ALIASES = {
        "shape": ["a", "s", "c", "k", "df"],
        "lambda": ["lam"],
        "mu": ["loc", "mean"],
        "std": ["scale", "sigma"],
    }

    def __init__(self, statistic_instance, display_name: str | None = None):
        """Initialize an adapter for one statistic engine.

        Args:
            statistic_instance: Concrete statistic instance from pysatl-criterion.
            display_name: Optional criterion name exposed to the expert system.
        """
        name = display_name or statistic_instance.code()
        super().__init__(name=name)
        self.engine = statistic_instance

    def _set_engine_parameters(self, parameters: dict) -> None:
        """Apply hypothesis parameters to attributes exposed by the engine."""
        for parameter_name, parameter_value in parameters.items():
            potential_targets = [parameter_name] + self.PARAM_ALIASES.get(parameter_name, [])
            for target in potential_targets:
                if hasattr(self.engine, target):
                    setattr(self.engine, target, parameter_value)
                    break

    def calculate(self, data, dist, params) -> float:
        """Calculate one raw goodness-of-fit statistic.

        Args:
            data: Sorted one-dimensional empirical sample.
            dist: Candidate distribution adapter.
            params: Parameters returned by the distribution's ``fit`` method.

        Returns:
            The raw statistic value returned by the criterion engine.

        Raises:
            Exception: Propagates errors raised by parameter conversion, CDF
                evaluation, or the statistic engine.
        """
        criterion_rvs, engine_parameters = dist.prepare_criterion_input(data, params)
        self._set_engine_parameters(engine_parameters)

        sig = inspect.signature(self.engine.execute_statistic)
        params_in_method = sig.parameters

        needs_cdf = "cdf_vals" in params_in_method

        has_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in params_in_method.values()
        )

        try:
            if needs_cdf or has_kwargs:
                cdf_vals = dist.cdf(data, params)
                return self.engine.execute_statistic(rvs=criterion_rvs, cdf_vals=cdf_vals)
            return self.engine.execute_statistic(rvs=criterion_rvs)
        except Exception as error:
            logger.debug("Error executing %s: %s", self.name, error)
            raise
