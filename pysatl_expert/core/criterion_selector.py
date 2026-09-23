"""Abstract criterion selector interface module."""

from abc import ABC, abstractmethod

from pysatl_expert.core.criterion import AbstractCriterion


class AbstractCriterionSelector(ABC):
    """Base interface for dynamic selection of Goodness-of-Fit tests.

    Filters and selects statistical tests applicable to a given sample and candidate distribution.
    """

    @abstractmethod
    def get_applicable_criteria(self, data, distribution) -> list[AbstractCriterion]:
        """Determine and return GoF criteria suitable for the provided sample and distribution.

        Args:
            data (np.ndarray): Numerical sample used to assess size and range constraints.
            distribution (AbstractDistribution): Candidate distribution model being evaluated.

        Returns:
            list[AbstractCriterion]: Collection of applicable criterion instances ready
                for execution.
        """
        pass
