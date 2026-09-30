"""Abstract Goodness-of-Fit criterion interface module."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class AbstractCriterion(ABC):
    """Base interface for Goodness-of-Fit (GoF) statistical criteria.

    Standardizes how the system calculates the discrepancy between an empirical
    sample and a theoretical distribution.

    Attributes:
        name (str): Unique identifier of the criterion (e.g., 'KS', 'AD').
    """

    def __init__(self, name: str):
        """Initialize the criterion with its identifying name.

        Args:
            name (str): Unique name identifier for the statistical criterion.
        """
        self.name = name

    @abstractmethod
    def calculate(self, data: np.ndarray, dist: Any, params: dict) -> float:
        """Compute the fit score for a candidate distribution.

        Args:
            data (np.ndarray): Sorted numerical sample to evaluate.
            dist (Any): Candidate distribution object implementing CDF/PDF.
            params (dict): Estimated parameters returned by the distribution's fit() method.

        Returns:
            float: Calculated statistical score (e.g., distance or likelihood).
        """
        pass
