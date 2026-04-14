import inspect
import logging

from pysatl_expert.core.criterion import AbstractCriterion


logger = logging.getLogger(__name__)


class GenericCriterion(AbstractCriterion):
    """
    Adapter for integrating 'pysatl-criterion' engines into the expert system.

    This class decouples the statistical calculation logic from the pipeline.
    It performs:
    1. Parameter Normalization: Maps SciPy-style parameter names (e.g., 'shape')
       to specific engine attributes (e.g., 'a', 's', 'df') using internal aliases.
    2. Dynamic Introspection: Uses Python's 'inspect' module to determine if the
       target statistic requires a Cumulative Distribution Function (CDF). This
       ensures lazy evaluation, calculating the CDF only when necessary.

    Attributes:
        PARAM_ALIASES (dict): A map used to resolve naming discrepancies between
            distribution fitting results and GoF test requirements.
    """

    PARAM_ALIASES = {
        "shape": ["a", "s", "c", "k", "df"],
        "lambda": ["lam"],
        "mu": ["loc", "mean"],
        "std": ["scale", "sigma"],
    }

    def __init__(self, statistic_instance, display_name: str | None = None):
        name = display_name or statistic_instance.code()
        super().__init__(name=name)
        self.engine = statistic_instance

    def calculate(self, data, dist, params):
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
