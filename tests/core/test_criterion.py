from typing import Any

import numpy as np

from pysatl_expert.core.criterion import AbstractCriterion


class CriterionStub(AbstractCriterion):
    def calculate(self, data: np.ndarray, dist: Any, params: dict) -> float:
        return 0.5


def test_abstract_criterion_initialization():
    name = "test_criterion"
    criterion = CriterionStub(name=name)
    assert criterion.name == name


def test_abstract_criterion_calculate():
    criterion = CriterionStub("stub")
    result = criterion.calculate(np.array([1, 2]), None, {})
    assert result == 0.5


def test_abstract_criterion_coverage():
    class CritFullStub(AbstractCriterion):
        def calculate(self, data, dist, params):
            return super().calculate(data, dist, params)

    stub = CritFullStub("TestCrit")
    stub.calculate(np.array([1]), None, {})
    assert stub.name == "TestCrit"
