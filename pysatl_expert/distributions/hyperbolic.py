import numpy as np
import scipy.stats as st

from pysatl_expert.core.distribution import AbstractDistribution


class HyperbolicDistribution(AbstractDistribution):
    """Four-parameter Hyperbolic probability distribution (genhyperbolic with p=1).

    Defined by shape (alpha > 0), skewness (|beta| < alpha), scale (delta > 0),
    and location (mu) with support (-inf, inf).
    """

    def __init__(self):
        super().__init__(name="Hyperbolic", support=(-np.inf, np.inf))

    def fit(self, data: np.ndarray) -> dict:
        values = np.asarray(data, dtype=float)
        center = float(np.mean(values))
        reference_scale = float(np.std(values))
        if not np.isfinite(reference_scale) or reference_scale <= 0.0:
            raise ValueError("Hyperbolic fitting requires a non-constant finite sample")
        standardized = (values - center) / reference_scale
        _, a, b, loc, scale = st.genhyperbolic.fit(standardized, fp=1.0)
        delta = float(scale * reference_scale)
        mu = float(center + loc * reference_scale)
        return {
            "alpha": float(a / delta),
            "beta": float(b / delta),
            "delta": delta,
            "mu": mu,
        }

    def pdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        delta = float(params["delta"])
        return st.genhyperbolic.pdf(
            data,
            p=1.0,
            a=float(params["alpha"]) * delta,
            b=float(params["beta"]) * delta,
            loc=params["mu"],
            scale=delta,
        )

    def cdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        delta = float(params["delta"])
        return st.genhyperbolic.cdf(
            data,
            p=1.0,
            a=float(params["alpha"]) * delta,
            b=float(params["beta"]) * delta,
            loc=params["mu"],
            scale=delta,
        )

    def prepare_criterion_input(self, data: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
        """Standardize observations for canonical Hyperbolic criteria."""
        delta = float(params["delta"])
        observations = self._standardize_criterion_input(data, params["mu"], delta)
        return observations, {
            "alpha": float(params["alpha"]) * delta,
            "beta": float(params["beta"]) * delta,
            "delta": 1.0,
            "mu": 0.0,
        }
