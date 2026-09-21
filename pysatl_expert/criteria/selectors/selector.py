from collections.abc import Collection, Mapping, Sequence

from pysatl_expert.core.criterion_selector import AbstractCriterionSelector
from pysatl_expert.criteria.calculate.generic import GenericCriterion
from pysatl_expert.criteria.catalog import (
    CRITERIA_REGISTRY,
    CriterionSpec,
)


def normalize_distribution_name(name: str) -> str:
    """Normalize expert distribution names to catalog keys."""
    return name.lower().replace("_", "")


class CriterionSelector(AbstractCriterionSelector):
    """Create adapters for all or a model-selected subset of available criteria."""

    def __init__(
        self,
        registry: Mapping[str, Sequence[CriterionSpec]] | None = None,
        feature_names: Collection[str] | None = None,
    ):
        super().__init__()
        self.registry = CRITERIA_REGISTRY if registry is None else registry
        self.feature_names = (
            frozenset(feature_names) if feature_names is not None else None
        )
        self._validate_requested_features()
        self._cache: dict[str, tuple[GenericCriterion, ...]] = {}

    def _validate_requested_features(self) -> None:
        if self.feature_names is None:
            return
        available = {
            spec.feature_name
            for specs in self.registry.values()
            for spec in specs
        }
        requested = {name for name in self.feature_names if "__" in name}
        missing = sorted(requested.difference(available))
        if missing:
            raise ValueError(
                "Model requires unavailable criterion features: " + ", ".join(missing)
            )

    def get_applicable_criteria(self, _data, distribution) -> list[GenericCriterion]:
        """Return criterion adapters selected for a candidate distribution."""
        distribution_name = normalize_distribution_name(distribution.name)
        if distribution_name not in self._cache:
            specs = self.registry.get(distribution_name, ())
            selected = (
                spec
                for spec in specs
                if self.feature_names is None
                or spec.feature_name in self.feature_names
            )
            self._cache[distribution_name] = tuple(
                GenericCriterion(spec.statistic_class(), spec.short_code)
                for spec in selected
            )
        return list(self._cache[distribution_name])
