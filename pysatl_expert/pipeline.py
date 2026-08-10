import logging
from typing import Any

import numpy as np

from pysatl_expert.core.pipeline_components import PipelineComponents
from pysatl_expert.models.feature_vector import FeatureVector


logger = logging.getLogger(__name__)


class DistributionPipeline:
    """Orchestrates identification of the best-fitting distribution for empirical sample data.

    Attributes:
        components (PipelineComponents): Registry container of pluggable components.
    """

    def __init__(self, components: PipelineComponents):
        """Initialize the pipeline with component registry.

        Args:
            components (PipelineComponents): Component registry instance.
        """
        self.components = components

    def _pre_validate(self, data_min: float, data_max: float, distributions: list) -> list:
        """Filter candidate distributions based on theoretical domain support boundaries.

        Args:
            data_min (float): Minimum value observed in the sample.
            data_max (float): Maximum value observed in the sample.
            distributions (list[AbstractDistribution]): Collection of candidate distributions.

        Returns:
            list[AbstractDistribution]: Candidate models whose domain supports contain
                the observed sample range [data_min, data_max].
        """
        valid_distributions = []
        for dist in distributions:
            s_min, s_max = dist.support
            if data_min >= s_min and data_max <= s_max:
                valid_distributions.append(dist)
        return valid_distributions

    def _evaluate_sample(self, data: np.ndarray) -> tuple[FeatureVector, dict[str, Any]]:
        """Process a single sample to extract sample statistics and GoF scores.

        Args:
            data (np.ndarray): 1D array of sample values.

        Returns:
            tuple[FeatureVector, dict[str, Any]]: FeatureVector and estimated parameters.
        """
        data = np.sort(data)
        sample_stats = self.components.feature_extractor.calculate_sample_stats(data)

        valid_dists = self._pre_validate(
            sample_stats["min"], sample_stats["max"], self.components.distributions
        )

        candidates_scores: dict[str, dict[str, float]] = {
            d.name: {} for d in self.components.distributions
        }
        all_params: dict[str, Any] = {}

        for dist in valid_dists:
            try:
                params = dist.fit(data)
                all_params[dist.name] = params

                dist_criteria = self.components.criterion_selector.get_applicable_criteria(
                    data, dist
                )

                for criterion in dist_criteria:
                    try:
                        val = criterion.calculate(data, dist, params)
                        candidates_scores[dist.name][criterion.name] = val
                    except Exception as e:
                        logger.warning(
                            f"Criterion '{criterion.name}' failed for '{dist.name}': {e}"
                        )
            except Exception as e:
                logger.error(f"Failed to fit distribution '{dist.name}': {e}")

        fv = FeatureVector(sample_stats=sample_stats, candidates_scores=candidates_scores)
        return fv, all_params

    def identify_best(self, data: np.ndarray, n_bootstraps: int = 100):
        """
        Executes the identification pipeline with bootstrap aggregation.
        """
        base_fv, base_params = self._evaluate_sample(data)

        bootstrap_fvs = []
        if n_bootstraps > 0:
            n = len(data)
            for i in range(n_bootstraps):
                try:
                    resample = np.random.choice(data, size=n, replace=True)
                    fv, _ = self._evaluate_sample(resample)
                    bootstrap_fvs.append(fv)
                except Exception as e:
                    logger.warning(f"Bootstrap iteration {i} failed: {e}")

        report = self.components.strategy.predict_report(base_fv, bootstrap_fvs)

        if report.distribution_name in base_params:
            report.parameters = base_params[report.distribution_name]

        return report
