import numpy as np
import scipy.stats as st

from pysatl_expert.core.distribution import AbstractDistribution


class UniformDistribution(AbstractDistribution):
    """
    Two-parameter implementation of the Continuous Uniform distribution.

    Defined by boundary parameters 'a' (minimum) and 'b' (maximum).
    While its theoretical support is (-inf, inf) for the purpose of fitting,
    its actual probability mass is strictly constrained within [a, b].

    Mapping to SciPy: 'a' maps to 'loc', 'b' is derived as 'loc + scale'.
    """
    def __init__(self):
        super().__init__(name="Uniform", support=(-np.inf, np.inf))

    def fit(self, data: np.ndarray) -> dict:
        loc, scale = st.uniform.fit(data)
        return {"a": float(np.min(data)) - 1e-9, "b": float(np.max(data)) + 1e-9}

    def pdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        return st.uniform.pdf(data, loc=params["a"], scale=params["b"] - params["a"])

    def cdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        return st.uniform.cdf(data, loc=params["a"], scale=params["b"] - params["a"])
