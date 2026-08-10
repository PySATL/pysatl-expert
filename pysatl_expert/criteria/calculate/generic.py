import inspect
import logging

from pysatl_expert.core.criterion import AbstractCriterion


logger = logging.getLogger(__name__)


class GenericCriterion(AbstractCriterion):
    """Adapter for integrating 'pysatl-criterion' engines into the expert system.

    Decouples statistical calculation logic from the core pipeline by mapping parameter
    names and dynamically evaluating required statistics.

    Attributes:
        PARAM_ALIASES (dict[str, list[str]]): Map resolving parameter naming discrepancies.
        engine (Any): Underlying statistic calculation instance from pysatl-criterion.
    """

    PARAM_ALIASES = {
        "shape": ["a", "s", "c", "k", "df"],
        "lambda": ["lam"],
        "mu": ["loc", "mean"],
        "std": ["scale", "sigma"],
    }

    def __init__(self, statistic_instance, display_name: str | None = None):
        """Initialize the generic criterion adapter.

        Args:
            statistic_instance (Any): Engine instance from pysatl-criterion.
            display_name (str | None): Optional custom display name override.
        """
        name = display_name or statistic_instance.code()
        super().__init__(name=name)
        self.engine = statistic_instance

    def calculate(self, data, dist, params) -> float:
        """Compute the goodness-of-fit statistic for the given distribution and sample.

        Args:
            data (np.ndarray): Sorted 1D numerical sample data array.
            dist (AbstractDistribution): Candidate distribution instance.
            params (dict): Estimated parameter dictionary returned by dist.fit().

        Returns:
            float: Calculated statistical criterion value.

        Raises:
            Exception: Re-raises any execution exception after logging debug info.
        """
        for p_name, p_value in params.items():
            potential_targets = [p_name] + self.PARAM_ALIASES.get(p_name, [])
            for target in potential_targets:
                if hasattr(self.engine, target):
                    setattr(self.engine, target, p_value)
                    break

        sig = inspect.signature(self.engine.execute_statistic)
        params_in_method = sig.parameters

        needs_cdf = "cdf_vals" in params_in_method
        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params_in_method.values())

        try:
            if needs_cdf or has_kwargs:
                cdf_vals = dist.cdf(data, params)
                return self.engine.execute_statistic(rvs=data, cdf_vals=cdf_vals)
            else:
                return self.engine.execute_statistic(rvs=data)
        except Exception as e:
            logger.debug(f"Error execute {self.name}: {e}")
            raise e
