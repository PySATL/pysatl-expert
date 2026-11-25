import numpy as np
import scipy.stats as st

from pysatl_expert.core.distribution import AbstractDistribution


class ParetoDistribution(AbstractDistribution):
    """Fixed-origin Type I Pareto probability distribution.

    Defined by shape (alpha > 0) and scale (x_m > 0) with loc fixed at zero.
    Support is [x_m, inf) subset of (0, inf).
    """

    def __init__(self):
        super().__init__(name="Pareto", support=(0.0, np.inf))

    def fit(self, data: np.ndarray) -> dict:
        values = np.asarray(data, dtype=float)
        if np.any(values <= 0.0):
            raise ValueError("Pareto fitting requires strictly positive observations")
        shape, _, scale = st.pareto.fit(values, floc=0.0)
        data_min = float(np.min(values))
        scale_val = float(scale)
        if scale_val >= data_min:
            adjusted = data_min * (1.0 - 1e-9)
            scale_val = (
                adjusted if 0.0 < adjusted < data_min else float(np.nextafter(data_min, 0.0))
            )
        return {"shape": float(shape), "loc": 0.0, "scale": scale_val}

    def pdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        return st.pareto.pdf(data, b=params["shape"], loc=params["loc"], scale=params["scale"])

    def cdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        return st.pareto.cdf(data, b=params["shape"], loc=params["loc"], scale=params["scale"])

    def prepare_criterion_input(self, data: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
        """Standardize observations for canonical Pareto criteria."""
        observations = self._standardize_criterion_input(
            data,
            params["loc"],
            params["scale"],
            positive_support=True,
        )
        return observations, {"shape": float(params["shape"]), "scale": 1.0}
