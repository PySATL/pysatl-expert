from pysatl_expert.core.criterion_selector import AbstractCriterionSelector


class SelectorStub(AbstractCriterionSelector):
    def get_applicable_criteria(self, data, distribution):
        return ["mock_criterion"]


def test_abstract_criterion_selector_interface():
    selector = SelectorStub()
    result = selector.get_applicable_criteria(None, None)
    assert result == ["mock_criterion"]


def test_abstract_selector_coverage():
    class SelFullStub(AbstractCriterionSelector):
        def get_applicable_criteria(self, data, dist):
            return super().get_applicable_criteria(data, dist)

    stub = SelFullStub()
    stub.get_applicable_criteria(None, None)
