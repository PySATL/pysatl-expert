from abc import ABC, abstractmethod

from pysatl_expert.core.criterion import AbstractCriterion


class AbstractCriterionSelector(ABC):
    """
    Defines the interface for components responsible for the dynamic selection
    of statistical Goodness-of-Fit (GoF) tests.
    """

    @abstractmethod
    def get_applicable_criteria(self, data, distribution) -> list[AbstractCriterion]:
        """
        Determines and returns a list of GoF criteria suitable for the provided context.

        Args:
            data (np.ndarray): The numerical sample used to assess sample size
                and data range constraints.
            distribution (AbstractDistribution): The specific distribution model
                currently being evaluated.

        Returns:
            list[AbstractCriterion]: A collection of criterion objects ready
                to perform the 'calculate' operation.
        """
        pass
