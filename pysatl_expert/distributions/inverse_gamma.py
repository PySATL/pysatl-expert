import numpy as np
import scipy.stats as st

from pysatl_expert.core.distribution import AbstractDistribution


class InverseGammaDistribution(AbstractDistribution):
    """Fixed-origin Inverse Gamma probability distribution.

    Defined by shape (alpha > 0) and scale (beta > 0) with loc fixed at zero.
    Support is [0, inf).
    """

    def __init__(self):
        super().__init__(name="InverseGamma", support=(0.0, np.inf))

    def fit(self, data: np.ndarray) -> dict:
        values = np.asarray(data, dtype=float)
        if np.any(values <= 0.0):
            raise ValueError("InverseGamma fitting requires strictly positive observations")
        shape, _, scale = st.invgamma.fit(values, floc=0.0)
        return {"shape": float(shape), "loc": 0.0, "scale": float(scale)}

    def pdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        return st.invgamma.pdf(data, a=params["shape"], loc=params["loc"], scale=params["scale"])

    def cdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        return st.invgamma.cdf(data, a=params["shape"], loc=params["loc"], scale=params["scale"])

    def prepare_criterion_input(self, data: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
        """Standardize observations for canonical Inverse Gamma criteria."""
        observations = self._standardize_criterion_input(
            data,
            params["loc"],
            params["scale"],
            positive_support=True,
        )
        return observations, {"alpha": float(params["shape"]), "beta": 1.0}
