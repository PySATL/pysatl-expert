from pysatl_expert.core.criterion import AbstractCriterion


class GenericCriterion(AbstractCriterion):
    """
    Adapter class for external statistical engines (e.g., 'pysatl-criterion').

    Integrates third-party mathematical implementations into the system's
    'AbstractCriterion' interface, ensuring scalability without code duplication.

    Attributes:
        engine: Underlying statistic instance (KS, AD, etc.) from the external library.
        name: Criterion identifier resolved from the engine or custom display name.
    """

    def __init__(self, statistic_instance, display_name: str | None = None):
        """
        Wraps a concrete statistical engine.

        Args:
            statistic_instance: Low-level engine implementing 'execute_statistic()'.
            display_name: Optional override for the criterion's name.
        """
        name = display_name or statistic_instance.code()
        super().__init__(name=name)
        self.engine = statistic_instance

    def calculate(self, data, dist, params):
        """
        Computes the fit score by delegating math to the wrapped engine.
        Uses the candidate distribution's CDF as the theoretical basis.
        """
        cdf_vals = dist.cdf(data, params)
        return self.engine.execute_statistic(rvs=data, cdf_vals=cdf_vals)
