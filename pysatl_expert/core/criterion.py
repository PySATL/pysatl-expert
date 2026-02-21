from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class AbstractCriterion(ABC):
    """
    Base interface for Goodness-of-Fit (GoF) statistical criteria.

    Standardizes how the system calculates the discrepancy between an empirical
    sample and a theoretical distribution.

    Attributes:
        name (str): Unique identifier of the criterion (e.g., 'KS', 'AD').
    """

    def __init__(self, name: str):
        """
        Initializes the criterion with its identifying name.
        """
        self.name = name

    @abstractmethod
    def calculate(self, data: np.ndarray, dist: Any, params: dict) -> float:
        """
        Computes the fit score for a candidate distribution.

        Args:
            data: Sorted numerical sample to evaluate.
            dist: Candidate distribution object (implements CDF/PDF).
            params: Parameters obtained from the distribution's fit() stage.

        Returns:
            float: Calculated statistical score (e.g., distance or likelihood).
        """
        pass
