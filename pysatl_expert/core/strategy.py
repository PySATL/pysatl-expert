from abc import ABC, abstractmethod

from pysatl_expert.models.feature_vector import FeatureVector
from pysatl_expert.models.report import Report


class AbstractStrategy(ABC):
    """
    Interface for the decision-making module of the expert system.

    The Strategy interprets statistical evidence (scores and sample stats)
    aggregated in a FeatureVector to select the most appropriate distribution.
    Allows for pluggable logic: from simple heuristics to ML-classifiers.
    """

    @abstractmethod
    def predict_report(
        self, base_fv: FeatureVector, bootstrap_fvs: list[FeatureVector] | None = None
    ) -> Report:
        """
        Analyzes the provided FeatureVectors to determine the most likely
        distribution law using original data and optional bootstrap results.

        Args:
            base_fv: FeatureVector calculated on the original sample.
            bootstrap_fvs: List of FeatureVectors calculated on resampled data.

        Returns:
            Report: The final verdict of the expert system.
        """
        pass
