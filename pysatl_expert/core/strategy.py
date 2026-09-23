"""Abstract decision strategy interface module."""

from abc import ABC, abstractmethod

from pysatl_expert.models.feature_vector import FeatureVector
from pysatl_expert.models.report import Report


class AbstractStrategy(ABC):
    """Base interface for the decision-making strategy module.

    Interprets aggregated sample statistics and GoF test scores to produce a distribution report.
    """

    @abstractmethod
    def predict_report(
        self, base_fv: FeatureVector, bootstrap_fvs: list[FeatureVector] | None = None
    ) -> Report:
        """Analyze FeatureVectors to determine the best-fitting distribution.

        Args:
            base_fv (FeatureVector): FeatureVector computed on original sample.
            bootstrap_fvs (list[FeatureVector] | None): Optional list of resampled FeatureVectors.

        Returns:
            Report: Final evaluation report with winner distribution and confidence scores.
        """
        pass
