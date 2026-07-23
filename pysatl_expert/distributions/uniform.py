"""Continuous Uniform probability distribution module."""

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
        loc, scale = st.uniform.fit(data)
        return {"a": float(np.min(data)) - 1e-9, "b": float(np.max(data)) + 1e-9}

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
