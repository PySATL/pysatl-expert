from unittest.mock import MagicMock

import numpy as np

from pysatl_expert.criteria.calculate.generic import GenericCriterion


def test_generic_criterion_init_auto_name():
    mock_engine = MagicMock()
    mock_engine.code.return_value = "KS_ENGINE"

    criterion = GenericCriterion(mock_engine)
    assert criterion.name == "KS_ENGINE"
    mock_engine.code.assert_called_once()


def test_generic_criterion_init_display_name():
    mock_engine = MagicMock()
    criterion = GenericCriterion(mock_engine, display_name="manual_name")
    assert criterion.name == "manual_name"


def test_generic_criterion_calculate(mocker):
    mock_engine = MagicMock()
    mock_engine.execute_statistic.return_value = 0.123
    mock_engine.code.return_value = "MOCK"

    mock_dist = MagicMock()
    mock_dist.cdf.return_value = np.array([0.1, 0.5, 0.9])

    criterion = GenericCriterion(mock_engine)

    data = np.array([1, 2, 3])
    params = {"mu": 0, "sigma": 1}

    result = criterion.calculate(data, mock_dist, params)

    assert result == 0.123

    mock_dist.cdf.assert_called_once_with(data, params)

    mock_engine.execute_statistic.assert_called_once()
    args, kwargs = mock_engine.execute_statistic.call_args
    assert np.array_equal(kwargs["rvs"], data)
    assert np.array_equal(kwargs["cdf_vals"], mock_dist.cdf.return_value)
