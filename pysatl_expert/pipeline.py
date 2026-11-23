from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import ArrayLike

from pysatl_expert import _procedure_execution
from pysatl_expert._procedure_execution import AnalysisPlan, ProcedureRegistry, Sample
from pysatl_expert.core.distribution import AbstractDistribution
from pysatl_expert.core.pipeline_components import PipelineComponents
from pysatl_expert.core.recommendation import (
    AnalysisProvider,
    ProcedureAdapter,
    ProcedureRequest,
    RecommendationMethod,
    StatisticProcedure,
)
from pysatl_expert.models.analysis_result import (
    AnalysisResult,
    ProcedureResult,
    RecommendationResult,
)


logger = logging.getLogger(__name__)


class _CatalogProcedure(Protocol):
    def evaluate(
        self, data: Sample, dist: AbstractDistribution, params: dict[str, Any]
    ) -> ProcedureResult: ...


class DistributionPipeline:
    """Orchestrate sample analysis and distribution recommendations.

    Attributes:
        components (PipelineComponents): Registry container of pluggable components.
    """

    def __init__(self, components: PipelineComponents) -> None:
        """Initialize the pipeline with component registry.

        Args:
            components (PipelineComponents): Component registry instance.
        """
        self.components = components

    @staticmethod
    def _validate_sample(data: ArrayLike) -> Sample:
        """Validate and normalize an empirical sample before statistical processing."""
        if np.ma.isMaskedArray(data) and np.any(np.ma.getmaskarray(data)):
            raise ValueError("Empirical sample must not contain masked observations")
        if np.iscomplexobj(data):
            raise ValueError("Empirical sample must contain only real values")
        values = np.asarray(data, dtype=float)
        if values.ndim != 1:
            raise ValueError("Empirical sample must be one-dimensional")
        if values.size == 0:
            raise ValueError("Empirical sample must be non-empty")
        if not np.all(np.isfinite(values)):
            raise ValueError("Empirical sample must contain only finite values")
        if np.ptp(values) == 0:
            raise ValueError("Empirical sample must be non-constant")
        return values

    def _pre_validate(
        self, data_min: float, data_max: float, distributions: list[AbstractDistribution]
    ) -> list[AbstractDistribution]:
        """Filter candidate distributions based on theoretical domain support boundaries.

        Args:
            data_min (float): Minimum value observed in the sample.
            data_max (float): Maximum value observed in the sample.
            distributions (list[AbstractDistribution]): Collection of candidate distributions.

        Returns:
            list[AbstractDistribution]: Candidate models whose domain supports contain
                the observed sample range [data_min, data_max].
        """
        valid_distributions = []
        for dist in distributions:
            s_min, s_max = dist.support
            if data_min >= s_min and data_max <= s_max:
                valid_distributions.append(dist)
        return valid_distributions

    def _analyze_sample(
        self,
        data: ArrayLike,
        requested_checks: dict[str, list[str]] | None = None,
        *,
        distributions: list[AbstractDistribution] | None = None,
    ) -> AnalysisResult:
        """Run the common analysis, optionally recording checks blocked by fit."""
        data = self._validate_sample(data)
        sample_snapshot = tuple(float(value) for value in data)
        data = np.sort(data)
        sample_stats = self.components.feature_extractor.calculate_sample_stats(data)

        distributions = self.components.distributions if distributions is None else distributions
        valid_dists = self._pre_validate(sample_stats["min"], sample_stats["max"], distributions)

        candidates_scores: dict[str, dict[str, float]] = {d.name: {} for d in distributions}
        all_params: dict[str, Any] = {}
        failures: dict[str, str] = {}
        procedure_results: dict[str, ProcedureResult] = {}
        catalog_ids: dict[str, dict[str, str]] = {d.name: {} for d in distributions}

        criterion_selector = self.components.criterion_selector

        for dist in distributions:
            checks = (requested_checks or {}).get(dist.name, [])
            if dist not in valid_dists:
                failures[f"fit:{dist.name}"] = "sample outside distribution support"
                failures.update(
                    {check: "not run: sample outside distribution support" for check in checks}
                )
                continue
            fit_succeeded = False
            params: dict[str, Any] = {}
            try:
                params = dist.fit(data)
                all_params[dist.name] = params
                fit_succeeded = True
            except Exception as e:
                reason = f"{type(e).__name__}: {e}"
                failures[f"fit:{dist.name}"] = reason
                failures.update({check: "not run: fit failed" for check in checks})
                logger.error(f"Failed to fit distribution '{dist.name}': {e}")
            try:
                if criterion_selector is None:
                    raise ValueError("Catalog analysis requires a criterion selector")
                dist_criteria = criterion_selector.get_applicable_criteria(data, dist)
            except Exception as e:
                reason = f"{type(e).__name__}: {e}"
                failures[f"selector:{dist.name}"] = reason
                failures.update({check: f"not run: selector failed: {reason}" for check in checks})
                logger.error(f"Failed to select criteria for '{dist.name}': {e}")
                continue
            for criterion in dist_criteria:
                key = f"{dist.name.lower().replace('_', '')}__{criterion.name.lower()}"
                try:
                    if not fit_succeeded and getattr(criterion, "requires_fit", True) is not False:
                        failures[key] = "not run: fit failed"
                        continue
                    if callable(getattr(type(criterion), "evaluate", None)):
                        result = cast(_CatalogProcedure, criterion).evaluate(data, dist, params)
                        if not isinstance(result, ProcedureResult):
                            raise TypeError("catalog procedure must return ProcedureResult")
                        if key in procedure_results:
                            raise ValueError(f"Duplicate catalog procedure ID: {key}")
                        procedure_results[key] = result
                        catalog_ids[dist.name][criterion.name] = key
                    else:
                        val = criterion.calculate(data, dist, params)
                        candidates_scores[dist.name][criterion.name] = (
                            _procedure_execution.finite_statistic(val)
                        )
                    failures.pop(key, None)
                except Exception as e:
                    failures[key] = f"{type(e).__name__}: {e}"
                    logger.warning(f"Criterion '{criterion.name}' failed for '{dist.name}': {e}")

        return AnalysisResult(
            sample_stats,
            all_params,
            candidates_scores,
            failures,
            procedure_results,
            sample_snapshot=sample_snapshot,
            catalog_ids=catalog_ids,
        )

    def analyze_catalog(self, data: ArrayLike) -> AnalysisResult:
        """Return the configured distribution catalog analysis for one sample."""
        return self._analyze_sample(data)

    @staticmethod
    def _required_statistics(declared: object) -> tuple[str, ...]:
        """Validate foreign declarations and consume a one-use iterable once."""
        if not isinstance(declared, Iterable) or isinstance(declared, (str, bytes)):
            raise ValueError("Required statistics must be a collection of statistic IDs")
        required: list[str] = []
        for name in declared:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Statistic IDs must be non-empty strings")
            required.append(name)
        if len(set(required)) != len(required):
            raise ValueError("Duplicate statistic IDs are not allowed")
        return tuple(required)

    @staticmethod
    def _validate_procedure_requests(
        required: tuple[str, ...],
        procedure_requests: Mapping[str, ProcedureRequest] | None,
    ) -> dict[str, ProcedureRequest]:
        requests = dict(procedure_requests or {})
        if not set(requests).issubset(required):
            raise ValueError("Procedure request must refer to a required statistic")
        if any(not isinstance(request, ProcedureRequest) for request in requests.values()):
            raise ValueError("Procedure requests must be ProcedureRequest instances")
        return requests

    @classmethod
    def _analysis_requirements(
        cls,
        declared: Iterable[str],
        procedures: ProcedureRegistry,
        procedure_requests: Mapping[str, ProcedureRequest] | None,
        allow_calibration: bool,
    ) -> AnalysisPlan:
        """Materialize and validate analysis declarations before computation."""
        required = cls._required_statistics(declared)
        validated_requests = cls._validate_procedure_requests(required, procedure_requests)
        if not isinstance(allow_calibration, bool):
            raise ValueError("allow_calibration must be a boolean")
        missing = set(required).difference(procedures)
        if missing:
            raise ValueError("Unavailable statistics: " + ", ".join(sorted(missing)))
        for name in required:
            operation = "evaluate" if name in validated_requests else "calculate"
            if not callable(getattr(procedures[name], operation, None)):
                raise ValueError(f"Procedure {name!r} must provide callable {operation}")
        return AnalysisPlan(
            required,
            validated_requests,
            {
                name: cast(StatisticProcedure, procedures[name])
                for name in required
                if name not in validated_requests
            },
            {
                name: cast(ProcedureAdapter, procedures[name])
                for name in required
                if name in validated_requests
            },
        )

    @classmethod
    def _method_requirements(
        cls,
        method: RecommendationMethod,
        procedures: ProcedureRegistry,
        allow_calibration: bool,
    ) -> AnalysisPlan:
        """Materialize and validate one method's declarations before computation."""
        declared = cls._required_statistics(getattr(method, "required_statistics", None))
        return cls._analysis_requirements(
            declared,
            procedures,
            getattr(method, "procedure_requests", {}),
            allow_calibration,
        )

    def _calculate_method_analysis(
        self,
        sample: Sample,
        requirements: AnalysisPlan,
        allow_calibration: bool,
        analysis: AnalysisResult | None = None,
    ) -> tuple[AnalysisResult, str | None]:
        if analysis is None:
            analysis = self._analyze_sample(sample, distributions=[])
        return _procedure_execution.execute_procedures(
            sample,
            requirements,
            allow_calibration,
            analysis,
        )

    def _prepare_method_analysis(
        self,
        data: ArrayLike,
        method: RecommendationMethod,
        procedures: ProcedureRegistry,
        allow_calibration: bool,
        analysis_provider: AnalysisProvider | None = None,
    ) -> tuple[Sample, AnalysisResult, str | None]:
        requirements = self._method_requirements(method, procedures, allow_calibration)
        sample = np.array(self._validate_sample(data), copy=True)
        analysis, error = self._seed_analysis(sample, analysis_provider)
        if analysis is not None and error is not None:
            return sample, analysis, error
        analysis, error = self._calculate_method_analysis(
            sample, requirements, allow_calibration, analysis
        )
        return sample, analysis, error

    def _seed_analysis(
        self, sample: Sample, analysis_provider: AnalysisProvider | None
    ) -> tuple[AnalysisResult | None, str | None]:
        """Seed a method flow once, isolating provider-owned values from mutation."""
        if analysis_provider is None:
            return None, None
        if not callable(analysis_provider):
            raise ValueError("Analysis provider must be callable")
        try:
            analysis = analysis_provider(sample.copy())
            if not isinstance(analysis, AnalysisResult):
                raise TypeError("analysis provider must return AnalysisResult")
            expected_snapshot = tuple(float(value) for value in sample)
            if analysis.sample_snapshot is None:
                raise ValueError("analysis provider must return a sample snapshot")
            if not isinstance(analysis.sample_snapshot, tuple):
                raise ValueError("analysis provider sample snapshot must be a tuple")
            if tuple(analysis.sample_snapshot) != expected_snapshot:
                raise ValueError("analysis provider snapshot does not match the requested sample")
            return deepcopy(analysis), None
        except Exception as exc:
            values = np.sort(sample)
            stats = self.components.feature_extractor.calculate_sample_stats(values)
            return (
                AnalysisResult(
                    stats,
                    {},
                    {},
                    {},
                    sample_snapshot=tuple(float(value) for value in sample),
                ),
                f"{type(exc).__name__}: {exc}",
            )

    def analyze_procedures(
        self,
        data: ArrayLike,
        *,
        required_statistics: Iterable[str],
        procedures: Mapping[str, StatisticProcedure | ProcedureAdapter],
        procedure_requests: Mapping[str, ProcedureRequest] | None = None,
        allow_calibration: bool = False,
    ) -> AnalysisResult:
        """Calculate explicitly requested procedure evidence without a method.

        Returns the analysis record even when a requested calculation or
        calibration is unavailable. Invalid declarations and unavailable
        procedure implementations raise before sample calculations begin.
        """
        procedures = dict(procedures)
        requirements = self._analysis_requirements(
            required_statistics,
            procedures,
            procedure_requests,
            allow_calibration,
        )
        sample = np.array(self._validate_sample(data), copy=True)
        analysis, _ = self._calculate_method_analysis(sample, requirements, allow_calibration)
        return analysis

    @staticmethod
    def _make_recommendation(
        sample: Sample,
        analysis: AnalysisResult,
        error: str | None,
        predict: Callable[[Sample, AnalysisResult], Any] | None,
    ) -> RecommendationResult:
        if error is not None:
            return RecommendationResult(analysis, error=error)
        try:
            if not callable(predict):
                raise TypeError("Recommendation method must provide callable predict")
            prediction = predict(sample.copy(), deepcopy(analysis))
        except Exception as exc:
            return RecommendationResult(analysis, error=f"{type(exc).__name__}: {exc}")
        return RecommendationResult(analysis, prediction=prediction)

    def recommend(
        self,
        data: ArrayLike,
        *,
        method: RecommendationMethod | None = None,
        procedures: Mapping[str, StatisticProcedure | ProcedureAdapter] | None = None,
        allow_calibration: bool = False,
        analysis_provider: AnalysisProvider | None = None,
        n_bootstraps: int = 0,
        random_state: int | None = None,
    ) -> RecommendationResult:
        """Run method-required sample statistics and pass them to a ready method.

        The method defaults to ``self.components.method`` and exposes
        ``required_statistics`` and ``predict(sample, analysis)``.
        Procedures map those IDs to ``StatisticProcedure`` or ``ProcedureAdapter``
        objects. Values are raw statistics unless calibration is requested and enabled.
        """
        self._validate_bootstrap_count(n_bootstraps)
        active_method = self.components.method if method is None else method
        if active_method is None:
            raise ValueError("Recommendation requires a method")
        active_procedures: ProcedureRegistry = {} if procedures is None else dict(procedures)
        sample, analysis, error = self._prepare_method_analysis(
            data,
            active_method,
            active_procedures,
            allow_calibration,
            analysis_provider,
        )
        result = self._make_recommendation(
            sample, analysis, error, getattr(active_method, "predict", None)
        )
        if n_bootstraps > 0:
            self._add_bootstrap_diagnostics(
                result,
                sample,
                active_method,
                active_procedures,
                allow_calibration,
                analysis_provider,
                n_bootstraps,
                random_state,
            )
        return result

    @staticmethod
    def _validate_bootstrap_count(n_bootstraps: int) -> None:
        if isinstance(n_bootstraps, bool) or not isinstance(n_bootstraps, int):
            raise ValueError("n_bootstraps must be a non-negative integer")
        if n_bootstraps < 0:
            raise ValueError("n_bootstraps must be a non-negative integer")

    @staticmethod
    def _prediction_label(prediction: object) -> str | None:
        if isinstance(prediction, Mapping):
            label = prediction.get("label")
            return label if isinstance(label, str) and label else None
        return prediction if isinstance(prediction, str) and prediction else None

    def _add_bootstrap_diagnostics(
        self,
        result: RecommendationResult,
        sample: Sample,
        method: RecommendationMethod,
        procedures: ProcedureRegistry,
        allow_calibration: bool,
        analysis_provider: AnalysisProvider | None,
        n_bootstraps: int,
        random_state: int | None,
    ) -> None:
        rng = np.random.default_rng(random_state)
        requested_label = self._prediction_label(result.prediction)
        predictions: list[str] = []
        failures: list[str] = []
        if result.error is not None or requested_label is None:
            reason = result.error or "primary prediction has no string label"
            failures.append(f"Primary recommendation unavailable: {reason}")
        else:
            for index in range(n_bootstraps):
                try:
                    resample = rng.choice(sample, size=len(sample), replace=True)
                    boot_sample, boot_analysis, error = self._prepare_method_analysis(
                        resample,
                        method,
                        procedures,
                        allow_calibration,
                        analysis_provider,
                    )
                    boot_result = self._make_recommendation(
                        boot_sample,
                        boot_analysis,
                        error,
                        getattr(method, "predict", None),
                    )
                    label = self._prediction_label(boot_result.prediction)
                    if boot_result.error is not None:
                        raise RuntimeError(boot_result.error)
                    if label is None:
                        raise ValueError("resample prediction has no string label")
                    predictions.append(label)
                except Exception as exc:
                    failures.append(
                        f"Bootstrap iteration {index} failed: {type(exc).__name__}: {exc}"
                    )
                    logger.warning(failures[-1])
        counts: dict[str, int] = {}
        for label in predictions:
            counts[label] = counts.get(label, 0) + 1
        successful = len(predictions)
        shares = (
            {label: count / successful for label, count in counts.items()} if successful else {}
        )
        result.diagnostics["bootstrap"] = {
            "requested": n_bootstraps,
            "successful": successful,
            "vote_counts": counts,
            "vote_shares": shares,
            "stability": (
                shares.get(requested_label, 0.0) if successful and requested_label else None
            ),
            "failures": failures,
        }
