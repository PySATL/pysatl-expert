from types import SimpleNamespace

import numpy as np
import pytest

from pysatl_expert.core.pipeline_components import PipelineComponents
from pysatl_expert.criteria.catalog import CRITERIA_REGISTRY, CRITERIA_SPECS, CriterionSpec
from pysatl_expert.criteria.selectors.selector import CriterionSelector
from pysatl_expert.distributions.normal import NormalDistribution
from pysatl_expert.models.feature_extractor import FeatureExtractor
from pysatl_expert.models.feature_vector import FeatureVector
from pysatl_expert.models.hierarchical_model import HierarchicalExpertModel
from pysatl_expert.pipeline import DistributionPipeline
from pysatl_expert.strategy.ml_strategy import MLStrategy


def make_strategy():
    model = HierarchicalExpertModel({"Symmetric": ["Normal", "Student"]})
    model.feature_names = FeatureVector.FEATURE_NAMES
    model.stage1_features = ["normal__ks", "sample_size"]
    model.stage2_features = {"Symmetric": ["normal__ad", "sample_size"]}

    class Forest:
        def __init__(self, classes):
            self.classes_ = np.asarray(classes)

        def predict_proba(self, frame):
            if len(self.classes_) == 1:
                return np.ones((len(frame), 1))
            scores = 1.0 / (1.0 + frame["normal__ad"].to_numpy())
            return np.column_stack([scores, 1.0 - scores])

    model.stage1_model = Forest(["Symmetric"])
    model.stage2_models = {"Symmetric": Forest(["Normal", "Student"])}
    strategy = object.__new__(MLStrategy)
    strategy.model = model
    strategy._class_names = model.classes_.tolist()
    return strategy


def test_required_features_include_all_stages_and_remove_duplicates():
    strategy = make_strategy()
    assert strategy.required_features == frozenset(["normal__ks", "normal__ad", "sample_size"])


def test_unknown_model_feature_fails_before_calculation():
    strategy = make_strategy()
    strategy.model.stage2_features["Symmetric"].append("normal__unknown")
    with pytest.raises(ValueError, match="normal__unknown"):
        _ = strategy.required_features


def test_nonhierarchical_model_keeps_full_schema():
    strategy = object.__new__(MLStrategy)
    strategy.model = SimpleNamespace()
    assert strategy.required_features == frozenset(FeatureVector.FEATURE_NAMES)


def test_selector_does_not_instantiate_unselected_statistics():
    class Unselected:
        @staticmethod
        def short_code():
            return "unselected"

        def __init__(self):
            raise AssertionError("unselected engine must not be instantiated")

    ks = next(s.statistic_class for s in CRITERIA_SPECS if s.feature_name == "normal__ks")
    selector = CriterionSelector(
        registry={
            "normal": [
                CriterionSpec("normal", "ks", ks),
                CriterionSpec("normal", "unselected", Unselected),
            ]
        },
        feature_names={"normal__ks", "sample_size"},
    )
    assert [c.name for c in selector.get_applicable_criteria(None, NormalDistribution())] == ["ks"]


def test_selector_rejects_model_features_missing_from_installed_criterion():
    ks = next(s for s in CRITERIA_SPECS if s.feature_name == "normal__ks")

    with pytest.raises(ValueError, match="normal__missing"):
        CriterionSelector(
            registry={"normal": [ks]},
            feature_names={"normal__ks", "normal__missing"},
        )


def test_pipeline_calculates_only_selected_statistics_for_original_and_bootstraps(monkeypatch):
    from pysatl_expert.criteria.calculate.generic import GenericCriterion

    strategy = make_strategy()
    components = PipelineComponents(
        distributions=[NormalDistribution()],
        criterion_selector=CriterionSelector(feature_names=strategy.required_features),
        strategy=strategy,
        feature_extractor=FeatureExtractor(),
    )
    pipeline = DistributionPipeline(components)
    sample = np.random.default_rng(19).normal(size=100)
    calls = []
    original = GenericCriterion.calculate

    def record(criterion, *args):
        calls.append(criterion.name)
        return original(criterion, *args)

    monkeypatch.setattr(GenericCriterion, "calculate", record)
    selected = pipeline.identify_best(sample, n_bootstraps=2, random_state=42)
    assert len(calls) == 2 * 3
    assert np.isnan(selected.all_scores["Normal"]["sw"])


def test_dataset_selector_still_calculates_full_registry():
    selector = CriterionSelector()
    criteria = selector.get_applicable_criteria(None, NormalDistribution())
    assert len(criteria) == len(CRITERIA_REGISTRY["normal"])
