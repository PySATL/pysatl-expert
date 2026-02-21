from unittest.mock import MagicMock, patch

from pysatl_expert.criteria.selectors.simple_selector import SimpleCriterionSelector


@patch(
    "pysatl_expert.core.criterion_selector.AbstractCriterionSelector.__init__", return_value=None
)
@patch("pysatl_expert.criteria.calculate.generic.GenericCriterion")
@patch("pysatl_criterion.statistics.normal.AndersonDarlingNormalityGofStatistic")
@patch("pysatl_criterion.statistics.normal.ShapiroWilkNormalityGofStatistic")
@patch("pysatl_criterion.statistics.exponent.GiniExponentialityGofStatistic")
@patch("pysatl_criterion.statistics.exponent.MoranExponentialityGofStatistic")
@patch("pysatl_criterion.statistics.weibull.AndersonDarlingWeibullGofStatistic")
@patch("pysatl_criterion.statistics.weibull.TikuSinghWeibullGofStatistic")
def test_simple_criterion_selector_init(
    mock_tiku,
    mock_ad_weibull,
    mock_moran,
    mock_gini,
    mock_shapiro,
    mock_ad_norm,
    mock_generic,
    mock_abstract_init,
):
    selector = SimpleCriterionSelector()

    assert "Normal" in selector._criteria_map
    assert "Exponential" in selector._criteria_map
    assert "Weibull" in selector._criteria_map
    assert selector._default_criteria == []
    assert len(selector._criteria_map["Normal"]) == 2
    assert len(selector._criteria_map["Exponential"]) == 2
    assert len(selector._criteria_map["Weibull"]) == 2


def test_get_applicable_criteria_normal():
    with patch(
        "pysatl_expert.core.criterion_selector.AbstractCriterionSelector.__init__",
        return_value=None,
    ):
        selector = SimpleCriterionSelector()
        mock_dist = MagicMock()
        mock_dist.name = "Normal"

        criteria = selector.get_applicable_criteria(None, mock_dist)

        assert len(criteria) == 2
        assert criteria == selector._criteria_map["Normal"]


def test_get_applicable_criteria_exponential():
    with patch(
        "pysatl_expert.core.criterion_selector.AbstractCriterionSelector.__init__",
        return_value=None,
    ):
        selector = SimpleCriterionSelector()
        mock_dist = MagicMock()
        mock_dist.name = "Exponential"

        criteria = selector.get_applicable_criteria(None, mock_dist)

        assert len(criteria) == 2
        assert criteria == selector._criteria_map["Exponential"]


def test_get_applicable_criteria_weibull():
    with patch(
        "pysatl_expert.core.criterion_selector.AbstractCriterionSelector.__init__",
        return_value=None,
    ):
        selector = SimpleCriterionSelector()
        mock_dist = MagicMock()
        mock_dist.name = "Weibull"

        criteria = selector.get_applicable_criteria(None, mock_dist)

        assert len(criteria) == 2
        assert criteria == selector._criteria_map["Weibull"]


def test_get_applicable_criteria_unknown():
    with patch(
        "pysatl_expert.core.criterion_selector.AbstractCriterionSelector.__init__",
        return_value=None,
    ):
        selector = SimpleCriterionSelector()
        mock_dist = MagicMock()
        mock_dist.name = "UnknownDistribution"

        criteria = selector.get_applicable_criteria(None, mock_dist)

        assert criteria == []
        assert criteria == selector._default_criteria
