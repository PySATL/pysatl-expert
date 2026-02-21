from abc import ABC, abstractmethod

import numpy as np


class AbstractDistribution(ABC):
    """
    Base interface for statistical distributions within the expert system.

    Standardizes parameter estimation and probability calculations. The 'support'
    attribute allows the pipeline to perform domain-based pre-validation.

    Attributes:
        _name (str): Unique identifier of the distribution.
        _support (tuple): Theoretical domain (min, max) of the probability function.
    """

    def __init__(self, name: str, support: tuple):
        """
        Initializes the distribution with its identity and domain constraints.
        """
        self._name = name
        self._support = support

    @property
    def name(self) -> str:
        """Name identifier of the distribution."""
        return self._name

    @property
    def support(self) -> tuple:
        """Theoretical domain (support) used for early-fail validation."""
        return self._support

    @abstractmethod
    def fit(self, data: np.ndarray) -> dict:
        """
        Estimates distribution parameters (MLE) from the provided sample.

        Returns:
            dict: Map of estimated parameters (e.g., {'mu': 0, 'std': 1}).
        """
        pass

    @abstractmethod
    def pdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        """
        Evaluates the Probability Density Function (PDF) at given points.
        """
        pass

    @abstractmethod
    def cdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        """
        Evaluates the Cumulative Distribution Function (CDF).
        The result serves as the basis for Goodness-of-Fit (GoF) calculations.
        """
        pass
