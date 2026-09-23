import numpy as np
import scipy.stats as st

from pysatl_expert.core.distribution import AbstractDistribution


class UniformDistribution(AbstractDistribution):
    """Two-parameter implementation of the Continuous Uniform distribution.

    Defined by lower boundary 'a' and upper boundary 'b' derived from sample min/max.

    Mapping to SciPy: 'a' maps to 'loc', 'b' is derived from 'loc + scale'.
    """

    def __init__(self):
        """Initialize the Uniform distribution with support (-inf, inf)."""
        super().__init__(name="Uniform", support=(-np.inf, np.inf))

    def fit(self, data: np.ndarray) -> dict:
        """Estimate boundary parameters 'a' (minimum) and 'b' (maximum).

        Args:
            data (np.ndarray): 1D array of sample observations.

        Returns:
            dict[str, float]: Map containing boundary parameters 'a' and 'b'.
        """
        values = np.asarray(data, dtype=float)
        data_min = float(np.min(values))
        data_max = float(np.max(values))
        span = data_max - data_min
        if not np.isfinite(span) or span <= 0:
            raise ValueError("Uniform fitting requires a non-constant finite sample")
        margin = span * 1e-9
        lower = data_min - margin
        if lower >= data_min:
            lower = float(np.nextafter(data_min, -np.inf))
        upper = data_max + margin
        if upper <= data_max:
            upper = float(np.nextafter(data_max, np.inf))
        return {"a": lower, "b": upper}

    def pdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Evaluate the Uniform probability density function (PDF).

        Args:
            data (np.ndarray): Array of values at which to evaluate the PDF.
            params (dict): Estimated parameter dictionary with 'a' and 'b'.

        Returns:
            np.ndarray: Computed PDF values.
        """
        return st.uniform.pdf(data, loc=params["a"], scale=params["b"] - params["a"])

    def cdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Evaluate the Uniform cumulative distribution function (CDF).

        Args:
            data (np.ndarray): Array of values at which to evaluate the CDF.
            params (dict): Estimated parameter dictionary with 'a' and 'b'.

        Returns:
            np.ndarray: Computed CDF values.
        """
        return st.uniform.cdf(data, loc=params["a"], scale=params["b"] - params["a"])

    def prepare_criterion_input(self, data: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
        """Map observations to the canonical Uniform interval."""
        lower = float(params["a"])
        observations = self._standardize_criterion_input(data, lower, float(params["b"]) - lower)
        return observations, {"a": 0.0, "b": 1.0}
