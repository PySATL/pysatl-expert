"""Pipeline components registry module."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pysatl_expert.core.criterion_selector import AbstractCriterionSelector
    from pysatl_expert.core.distribution import AbstractDistribution
    from pysatl_expert.core.recommendation import RecommendationMethod
    from pysatl_expert.models.feature_extractor import FeatureExtractor


class PipelineComponents:
    """Registry container aggregating modular components for the expert pipeline.

    Centralizes candidate distributions, criterion selection logic, and feature extraction.

    Attributes:
        distributions (list[AbstractDistribution]): Candidate statistical models.
        criterion_selector (AbstractCriterionSelector): Logic for picking applicable GoF tests.
        method (RecommendationMethod): Optional primary recommendation method.
        feature_extractor (FeatureExtractor): Service for extracting continuous sample statistics.
    """

    def __init__(
        self,
        distributions: list[AbstractDistribution],
        criterion_selector: AbstractCriterionSelector | None,
        method: RecommendationMethod | None,
        feature_extractor: FeatureExtractor,
    ):
        """Initialize the pipeline component registry.

        Args:
            distributions (list[AbstractDistribution]): Candidate statistical model instances.
            criterion_selector (AbstractCriterionSelector): GoF test selector instance.
            method (RecommendationMethod): Optional primary recommendation method.
            feature_extractor (FeatureExtractor): Feature extraction service instance.
        """
        self.distributions = distributions
        self.criterion_selector = criterion_selector
        self.method = method
        self.feature_extractor = feature_extractor
