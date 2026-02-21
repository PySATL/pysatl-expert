from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pysatl_expert.models.feature_vector import FeatureVector
from pysatl_expert.pipeline import DistributionPipeline


class TestDistributionPipeline:
    @pytest.fixture
    def mock_components(self):
        components = MagicMock()

        components.feature_extractor.calculate_sample_stats.return_value = {
            "min": 0.0,
            "max": 10.0,
            "mean": 5.0,
        }

        dist1 = MagicMock()
        dist1.name = "normal"
        dist1.support = (-float("inf"), float("inf"))
        dist1.fit.return_value = (0, 1)

        dist2 = MagicMock()
        dist2.name = "beta"
        dist2.support = (0, 1)

        components.distributions = [dist1, dist2]

        criterion = MagicMock()
        criterion.name = "AIC"
        criterion.calculate.return_value = 10.5
        components.criterion_selector.get_applicable_criteria.return_value = [criterion]

        report = MagicMock()
        report.distribution_name = "normal"
        report.parameters = None
        components.strategy.predict_report.return_value = report

        return components

    def test_pre_validate(self, mock_components):
        pipeline = DistributionPipeline(mock_components)

        dist_in = MagicMock(support=(0, 10))
        dist_out_min = MagicMock(support=(1, 10))
        dist_out_max = MagicMock(support=(0, 9))

        dists = [dist_in, dist_out_min, dist_out_max]
        valid = pipeline._pre_validate(0, 10, dists)

        assert len(valid) == 1
        assert valid[0] == dist_in

    def test_evaluate_sample_success(self, mock_components):
        pipeline = DistributionPipeline(mock_components)
        data = np.array([10, 0, 5])

        fv, params = pipeline._evaluate_sample(data)

        assert isinstance(fv, FeatureVector)
        assert "normal" in params
        assert params["normal"] == (0, 1)
        assert fv.candidates_scores["normal"]["AIC"] == 10.5
        assert "beta" not in params

    def test_evaluate_sample_fit_exception(self, mock_components):
        mock_components.distributions[0].fit.side_effect = Exception("Fit error")

        pipeline = DistributionPipeline(mock_components)
        data = np.array([1, 2, 3])

        fv, params = pipeline._evaluate_sample(data)

        assert "normal" not in params
        assert fv.candidates_scores["normal"] == {}

    def test_evaluate_sample_criterion_exception(self, mock_components):
        criterion = mock_components.criterion_selector.get_applicable_criteria.return_value[0]
        criterion.calculate.side_effect = Exception("Calc error")

        pipeline = DistributionPipeline(mock_components)
        data = np.array([1, 2, 3])

        fv, params = pipeline._evaluate_sample(data)

        assert "normal" in params
        assert "AIC" not in fv.candidates_scores["normal"]

    def test_identify_best_with_bootstrap(self, mock_components):
        pipeline = DistributionPipeline(mock_components)
        data = np.array([1, 2, 3, 4, 5])

        report = pipeline.identify_best(data, n_bootstraps=2)

        assert mock_components.strategy.predict_report.called
        args, _ = mock_components.strategy.predict_report.call_args
        assert len(args[1]) == 2
        assert report.parameters == (0, 1)

    def test_identify_best_no_bootstrap(self, mock_components):
        pipeline = DistributionPipeline(mock_components)
        data = np.array([1, 2, 3])

        report = pipeline.identify_best(data, n_bootstraps=0)

        args, _ = mock_components.strategy.predict_report.call_args
        assert len(args[1]) == 0
        assert report.parameters == (0, 1)

    def test_identify_best_params_not_found(self, mock_components):
        report_mock = MagicMock()
        report_mock.distribution_name = "unknown_dist"
        report_mock.parameters = None
        mock_components.strategy.predict_report.return_value = report_mock

        pipeline = DistributionPipeline(mock_components)
        data = np.array([1, 2, 3])

        report = pipeline.identify_best(data, n_bootstraps=0)

        assert report.parameters is None

    def test_identify_best_bootstrap_exception(self, mock_components):
        pipeline = DistributionPipeline(mock_components)
        data = np.array([1, 2, 3])

        with patch.object(pipeline, "_evaluate_sample") as mocked_eval:
            mocked_eval.side_effect = [
                (MagicMock(), {"normal": (0, 1)}),
                Exception("Bootstrap iteration failed"),
            ]

            report = pipeline.identify_best(data, n_bootstraps=1)

            assert mocked_eval.call_count == 2
            assert report.distribution_name == "normal"

    def test_evaluate_sample_no_valid_distributions(self, mock_components):
        for dist in mock_components.distributions:
            dist.support = (100, 200)

        pipeline = DistributionPipeline(mock_components)
        data = np.array([1, 2, 3])

        fv, params = pipeline._evaluate_sample(data)

        assert params == {}
        assert fv.candidates_scores["normal"] == {}
        assert fv.candidates_scores["beta"] == {}
