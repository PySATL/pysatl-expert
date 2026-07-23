"""Abstract distribution interface module."""

from abc import ABC, abstractmethod

import numpy as np


class AbstractDistribution(ABC):
    """Base interface for statistical probability distributions within the expert system.

    Standardizes parameter estimation and probability calculations across candidate models.

    Attributes:
        name (str): Unique identifier of the distribution.
        support (tuple[float, float]): Theoretical domain (min, max) used for pre-validation.
    """

    def __init__(self, name: str, support: tuple):
        """Initialize the distribution with identity and theoretical domain bounds.

        Args:
            name (str): Unique identifier name of the distribution.
            support (tuple[float, float]): Theoretical domain tuple (min, max).
        """
        self._name = name
        self._support = support

    @property
    def name(self) -> str:
        """Get the name identifier of the distribution."""
        return self._name

    @property
    def support(self) -> tuple:
        """Get the theoretical domain (support) tuple used for pre-validation."""
        return self._support

    @abstractmethod
    def fit(self, data: np.ndarray) -> dict:
        """Estimate distribution parameters via Maximum Likelihood Estimation (MLE).

        Args:
            data (np.ndarray): 1D array of sample observations.

        Returns:
            dict[str, float]: Map of estimated parameter names to float values.
        """
        pass

    @abstractmethod
    def pdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Evaluate the Probability Density Function (PDF) for sample points.

        Args:
            data (np.ndarray): Array of values at which to evaluate the PDF.
            params (dict): Estimated parameter dictionary returned by fit().

        Returns:
            np.ndarray: Computed PDF density values.
        """
        pass

    @abstractmethod
    def cdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Evaluate the Cumulative Distribution Function (CDF) for sample points.

        Args:
            data (np.ndarray): Array of values at which to evaluate the CDF.
            params (dict): Estimated parameter dictionary returned by fit().

        Returns:
            np.ndarray: Computed CDF probability values.
        """
        pass
