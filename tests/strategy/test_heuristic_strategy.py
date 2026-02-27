from unittest.mock import MagicMock

import pytest

from pysatl_expert.models.feature_vector import FeatureVector
from pysatl_expert.strategy.heuristic_strategy import HeuristicStrategy


@pytest.fixture
def strategy():
    return HeuristicStrategy()


def create_mock_fv(scores_dict):
    fv = MagicMock(spec=FeatureVector)
    fv.candidates_scores = scores_dict
    return fv


class TestHeuristicStrategyFullCoverage:
    def test_penalty_normal_ad(self, strategy):
        scores = {"anderson_darling": 0.5}
        assert strategy._calculate_penalty("normal", scores) == 0.5

    def test_penalty_normal_shapiro(self, strategy):
        scores = {"shapiro_wilk": 0.9}
        assert pytest.approx(strategy._calculate_penalty("normal", scores)) == 3.0

    def test_penalty_expon_gini(self, strategy):
        scores = {"gini_index": 0.4}
        assert pytest.approx(strategy._calculate_penalty("expon", scores)) == 2.0

    def test_penalty_expon_moran(self, strategy):
        scores = {"moran_test": 0.15}
        assert strategy._calculate_penalty("expon", scores) == 0.15

    def test_penalty_weibull_ad(self, strategy):
        scores = {"anderson_darling": 0.5}
        assert pytest.approx(strategy._calculate_penalty("weibull", scores)) == 0.7

    def test_penalty_weibull_tiku_singh(self, strategy):
        scores = {"tiku_singh": 0.8}
        assert pytest.approx(strategy._calculate_penalty("weibull", scores)) == 2.2

    def test_penalty_fallbacks(self, strategy):
        assert strategy._calculate_penalty("unknown", {}) == 100.0
        assert strategy._calculate_penalty("normal", {"random": 1}) == 100.0
        assert strategy._calculate_penalty("weibull", {"random": 1}) == 100.2

    def test_calculate_penalty_debug(self, strategy, capsys):
        strategy._calculate_penalty("normal", {"anderson_darling": 0.5}, debug=True)
        captured = capsys.readouterr()
        assert "[DEBUG]" in captured.out

    def test_choose_winner_no_active(self, strategy):
        fv = create_mock_fv({})
        assert strategy._choose_winner_from_fv(fv) == "None"

    def test_choose_winner_with_debug(self, strategy, capsys):
        fv = create_mock_fv({"normal": {"anderson_darling": 0.5}})
        strategy._choose_winner_from_fv(fv, debug=True)
        captured = capsys.readouterr()
        assert "------------------------------------------------------------" in captured.out

    def test_predict_report_no_bootstrap(self, strategy):
        fv = create_mock_fv({"normal": {"anderson_darling": 0.5}})
        report = strategy.predict_report(fv, bootstrap_fvs=None)
        assert report.distribution_name == "normal"
        assert report.confidence == 0.0

    def test_predict_report_empty_votes(self, strategy):
        base_fv = create_mock_fv({"normal": {"anderson_darling": 0.5}})
        boot_fv = create_mock_fv({})

        report = strategy.predict_report(base_fv, bootstrap_fvs=[boot_fv])
        assert report.distribution_name == "None"
        assert report.confidence == 0.0

    def test_predict_report_full_success(self, strategy):
        base_fv = create_mock_fv(
            {"normal": {"anderson_darling": 0.5}, "expon": {"gini_index": 0.5}}
        )

        boot1 = create_mock_fv({"normal": {"anderson_darling": 0.1}})
        boot2 = create_mock_fv({"normal": {"anderson_darling": 0.1}})
        boot3 = create_mock_fv({"expon": {"gini_index": 0.5}})

        report = strategy.predict_report(base_fv, bootstrap_fvs=[boot1, boot2, boot3])

        assert report.distribution_name == "normal"
        assert report.confidence == 0.67
        assert report.final_ranks["normal"] == 2
