from pysatl_expert.core.criterion_selector import AbstractCriterionSelector
from pysatl_expert.core.distribution import AbstractDistribution
from pysatl_expert.core.strategy import AbstractStrategy
from pysatl_expert.models.feature_extractor import FeatureExtractor


class PipelineComponents:
    """
    A registry container aggregating modular components for the expert system.

    Centralizes candidate distributions, selection logic, decision strategies,
    and profiling tools. This decoupling allows the pipeline logic to remain
    independent of specific mathematical or algorithmic implementations.

    Attributes:
        distributions (list[AbstractDistribution]): Candidate statistical models.
        criterion_selector (AbstractCriterionSelector): Logic for picking
            appropriate Goodness-of-Fit tests.
        strategy (AbstractStrategy): Module for final distribution selection
            (ML-based or heuristic).
        feature_extractor (FeatureExtractor): Service for intrinsic data profiling.
    """

    def __init__(
        self,
        distributions: list[AbstractDistribution],
        criterion_selector: AbstractCriterionSelector,
        strategy: AbstractStrategy,
        feature_extractor: FeatureExtractor,
    ):
        """
        Initializes the component registry with pluggable modules.
        """
        self.distributions = distributions
        self.criterion_selector = criterion_selector
        self.strategy = strategy
        self.feature_extractor = feature_extractor
