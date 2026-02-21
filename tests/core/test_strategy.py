from pysatl_expert.core.strategy import AbstractStrategy
from pysatl_expert.models.feature_vector import FeatureVector
from pysatl_expert.models.report import Report


class StrategyStub(AbstractStrategy):
    def predict_report(self, feature_vector: FeatureVector) -> Report:
        return Report("StubDist", 1.0, {})


def test_abstract_strategy_interface():
    strategy = StrategyStub()
    fv = FeatureVector({}, {})
    report = strategy.predict_report(fv)

    assert isinstance(report, Report)
    assert report.distribution_name == "StubDist"


def test_abstract_strategy_coverage():
    class StratFullStub(AbstractStrategy):
        def predict_report(self, fv):
            return super().predict_report(fv)

    stub = StratFullStub()
    stub.predict_report(None)
