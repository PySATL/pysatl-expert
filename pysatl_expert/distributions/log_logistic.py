import numpy as np
import scipy.stats as st

from pysatl_expert.core.distribution import AbstractDistribution


class LogLogisticDistribution(AbstractDistribution):
    """Fixed-origin Log-Logistic (Fisk) probability distribution.

    Defined by shape (beta > 0) and scale (alpha > 0) with loc fixed at zero.
    Support is [0, inf).
    """

    def __init__(self):
        super().__init__(name="LogLogistic", support=(0.0, np.inf))

    def fit(self, data: np.ndarray) -> dict:
        values = np.asarray(data, dtype=float)
        if np.any(values <= 0.0):
            raise ValueError("LogLogistic fitting requires strictly positive observations")
        shape, _, scale = st.fisk.fit(values, floc=0.0)
        return {"shape": float(shape), "loc": 0.0, "scale": float(scale)}

    def pdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        return st.fisk.pdf(data, c=params["shape"], loc=params["loc"], scale=params["scale"])

    def cdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        return st.fisk.cdf(data, c=params["shape"], loc=params["loc"], scale=params["scale"])

    def prepare_criterion_input(self, data: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
        """Standardize observations for canonical Log-Logistic criteria."""
        observations = self._standardize_criterion_input(
            data,
            params["loc"],
            params["scale"],
            positive_support=True,
        )
        return observations, {"alpha": 1.0, "beta": float(params["shape"])}
