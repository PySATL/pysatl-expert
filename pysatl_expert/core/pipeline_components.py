"""Pipeline components registry module."""

from pysatl_expert.core.criterion_selector import AbstractCriterionSelector
from pysatl_expert.core.distribution import AbstractDistribution
from pysatl_expert.core.strategy import AbstractStrategy
from pysatl_expert.models.feature_extractor import FeatureExtractor


class PipelineComponents:
    """Registry container aggregating modular components for the expert pipeline.

    Centralizes candidate distributions, criterion selection logic, decision strategy,
    and feature extraction tools.

    Attributes:
        distributions (list[AbstractDistribution]): Candidate statistical models.
        criterion_selector (AbstractCriterionSelector): Logic for picking applicable GoF tests.
        strategy (AbstractStrategy): Decision strategy for final distribution ranking.
        feature_extractor (FeatureExtractor): Service for extracting continuous sample statistics.
    """

    def __init__(
        self,
        distributions: list[AbstractDistribution],
        criterion_selector: AbstractCriterionSelector,
        strategy: AbstractStrategy,
        feature_extractor: FeatureExtractor,
    ):
        """Initialize the pipeline component registry.

        Args:
            distributions (list[AbstractDistribution]): Candidate statistical model instances.
            criterion_selector (AbstractCriterionSelector): GoF test selector instance.
            strategy (AbstractStrategy): Decision strategy instance.
            feature_extractor (FeatureExtractor): Feature extraction service instance.
        """
        self.distributions = distributions
        self.criterion_selector = criterion_selector
        self.strategy = strategy
        self.feature_extractor = feature_extractor
