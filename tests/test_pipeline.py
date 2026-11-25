"""Canonical pipeline workflow checks use controlled method/procedure doubles."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from pysatl_expert.core.recommendation import RecommendationMethod
from pysatl_expert.models.analysis_result import AnalysisResult
from pysatl_expert.pipeline import DistributionPipeline


class LabelMethod:
    required_statistics = ("mean",)

    def __init__(self):
        self.calls = []

    def predict(self, sample, analysis):
        self.calls.append((sample.copy(), analysis.sample_snapshot))
        label = "low" if float(np.mean(sample)) < 0.2 else "high"
        return {"label": label, "scores": {label: 1.0}}


def make_pipeline():
    components = SimpleNamespace(
        feature_extractor=MagicMock(), distributions=[], criterion_selector=None, method=None
    )
    components.feature_extractor.calculate_sample_stats.side_effect = lambda values: {
        "sample_size": len(values),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }
    return DistributionPipeline(components)


def test_catalog_analysis_keeps_original_order_snapshot():
    sample = np.array([0.8, 0.2, 0.4])
    result = make_pipeline().analyze_catalog(sample)

    assert result.sample_snapshot == (0.8, 0.2, 0.4)
    assert result.sample_stats["min"] == 0.2
    assert result.sample_stats["max"] == 0.8


@pytest.mark.parametrize(
    "sample, reason",
    [
        (np.array([1 + 9j, 2 + 8j, 4 + 7j]), "only real values"),
        (np.ma.array([1.0, 2.0, 1000.0], mask=[False, False, True]), "masked observations"),
    ],
)
def test_public_analysis_rejects_lossy_input_before_processing(sample, reason):
    from pysatl_expert.app import analyze_sample

    with pytest.raises(ValueError, match=reason):
        analyze_sample(sample, distributions=["Normal"], statistics=["normal__ks"])


def test_unmasked_real_array_keeps_all_observations():
    from pysatl_expert.app import analyze_sample

    result = analyze_sample(
        np.ma.array([1.0, 2.0, 4.0], mask=False),
        distributions=["Normal"],
        statistics=["normal__ks"],
    )
    assert result.sample_snapshot == (1.0, 2.0, 4.0)
    assert result.sample_stats["mean"] == pytest.approx(7 / 3)


def test_provider_requires_matching_original_order_snapshot():
    method = LabelMethod()
    sample = np.array([0.8, 0.2, 0.4])
    pipeline = make_pipeline()

    result = pipeline.recommend(
        sample,
        method=method,
        procedures={"mean": SimpleNamespace(calculate=lambda values: float(np.mean(values)))},
        analysis_provider=lambda values: AnalysisResult(
            {"sample_size": 3}, {}, {}, {}, sample_snapshot=tuple(sorted(values))
        ),
    )

    assert "snapshot does not match" in result.error
    assert method.calls == []


def test_bootstrap_uses_same_method_preserves_primary_and_records_votes(monkeypatch):
    sample = np.array([0.1, 0.2, 0.3])
    resamples = iter((np.array([0.1, 0.1, 0.3]), np.array([0.2, 0.3, 0.3])))
    rng = SimpleNamespace(choice=lambda *_args, **_kwargs: next(resamples))
    monkeypatch.setattr(np.random, "default_rng", lambda seed: rng)
    method = LabelMethod()
    procedure = MagicMock()
    procedure.calculate.side_effect = lambda values: float(np.mean(values))

    result = make_pipeline().recommend(
        sample,
        method=method,
        procedures={"mean": procedure},
        n_bootstraps=2,
        random_state=17,
    )

    assert result.prediction == {"label": "high", "scores": {"high": 1.0}}
    assert result.diagnostics["bootstrap"] == {
        "requested": 2,
        "successful": 2,
        "vote_counts": {"low": 1, "high": 1},
        "vote_shares": {"low": 0.5, "high": 0.5},
        "stability": 0.5,
        "failures": [],
    }
    assert len(method.calls) == 3
    np.testing.assert_array_equal(method.calls[0][0], sample)
    assert procedure.calculate.call_count == 3


def test_failed_resamples_are_counted_and_zero_votes_have_no_stability(monkeypatch):
    sample = np.array([0.1, 0.2, 0.3])
    rng = SimpleNamespace(choice=lambda *_args, **_kwargs: np.array([0.2, 0.3, 0.4]))
    monkeypatch.setattr(np.random, "default_rng", lambda seed: rng)

    class UnavailableResample(LabelMethod):
        def predict(self, values, analysis):
            if float(np.mean(values)) > 0.25:
                raise RuntimeError("technical stub failure")
            return super().predict(values, analysis)

    result = make_pipeline().recommend(
        sample,
        method=UnavailableResample(),
        procedures={"mean": SimpleNamespace(calculate=lambda values: float(np.mean(values)))},
        n_bootstraps=2,
    )

    diagnostics = result.diagnostics["bootstrap"]
    assert diagnostics["requested"] == 2
    assert diagnostics["successful"] == 0
    assert diagnostics["vote_counts"] == {}
    assert diagnostics["vote_shares"] == {}
    assert diagnostics["stability"] is None
    assert len(diagnostics["failures"]) == 2


def test_successful_votes_without_primary_label_have_zero_stability(monkeypatch):
    sample = np.array([0.1, 0.2, 0.3])
    rng = SimpleNamespace(choice=lambda *_args, **_kwargs: np.array([0.1, 0.1, 0.2]))
    monkeypatch.setattr(np.random, "default_rng", lambda seed: rng)
    result = make_pipeline().recommend(
        sample,
        method=LabelMethod(),
        procedures={"mean": SimpleNamespace(calculate=lambda values: float(np.mean(values)))},
        n_bootstraps=2,
    )
    diagnostics = result.diagnostics["bootstrap"]
    assert diagnostics["successful"] == 2
    assert diagnostics["vote_counts"] == {"low": 2}
    assert diagnostics["stability"] == 0.0


@pytest.mark.parametrize("count", [-1, True, 1.5, "2"])
def test_bootstrap_count_is_validated_before_method_execution(count):
    method = LabelMethod()
    with pytest.raises(ValueError, match="non-negative integer"):
        make_pipeline().recommend(
            [0.1, 0.2, 0.3],
            method=method,
            procedures={"mean": SimpleNamespace(calculate=lambda values: float(np.mean(values)))},
            n_bootstraps=count,
        )
    assert method.calls == []


def test_explicit_recommendation_protocol_subclass_can_initialize_requirements():
    class Ready(RecommendationMethod):
        def __init__(self):
            self.required_statistics = ["mean"]

        def predict(self, sample, analysis):
            return {"label": "ready", "mean": analysis.special_statistics["mean"]}

    result = make_pipeline().recommend(
        np.array([0.1, 0.2, 0.3]),
        method=Ready(),
        procedures={"mean": SimpleNamespace(calculate=lambda values: float(np.mean(values)))},
    )

    assert result.error is None
    assert result.prediction == {"label": "ready", "mean": pytest.approx(0.2)}
    assert result.analysis.sample_snapshot == (0.1, 0.2, 0.3)
