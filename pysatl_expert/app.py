from collections.abc import Iterable
from pathlib import Path

from pysatl_expert.core.pipeline_components import PipelineComponents
from pysatl_expert.criteria.catalog import CRITERIA_SPECS
from pysatl_expert.criteria.selectors.selector import CriterionSelector
from pysatl_expert.distributions import DISTRIBUTION_REGISTRY, create_all_distributions
from pysatl_expert.models.feature_extractor import FeatureExtractor
from pysatl_expert.pipeline import DistributionPipeline
from pysatl_expert.reporting.plot import generate_plot_report
from pysatl_expert.reporting.result_report import (
    ReportOptions,
    render_result_report,
    write_result_pdf,
)
from pysatl_expert.strategy.ml_strategy import MLStrategy


def analyze_sample(data, *, distributions, statistics):
    """Analyze a sample without a model using explicit distribution and statistic names.

    Return descriptive statistics, fitted parameters, raw statistics and failures.
    Each requested statistic must belong to one of the requested distributions.
    """
    values = DistributionPipeline._validate_sample(data)
    if isinstance(distributions, str) or isinstance(statistics, str):
        raise ValueError("distributions and statistics must be collections of names")
    names = list(distributions)
    checks = list(statistics)
    if not names or not checks or not all(isinstance(n, str) for n in names + checks):
        raise ValueError("distributions and statistics must be non-empty collections of names")
    unknown = set(names) - DISTRIBUTION_REGISTRY.keys()
    if unknown:
        raise ValueError(f"Unknown distributions: {', '.join(sorted(unknown))}")
    specs = {spec.feature_name: spec for spec in CRITERIA_SPECS}
    unknown_checks = set(checks) - specs.keys()
    if unknown_checks:
        raise ValueError(f"Unknown statistics: {', '.join(sorted(unknown_checks))}")
    requested_checks = {name: [] for name in dict.fromkeys(names)}
    by_key = {name.lower(): name for name in requested_checks}
    for check in dict.fromkeys(checks):
        key = specs[check].distribution
        if key not in by_key:
            raise ValueError(f"Statistic {check} requires distribution {key}")
        requested_checks[by_key[key]].append(check)
    components = PipelineComponents(
        distributions=[DISTRIBUTION_REGISTRY[name]() for name in requested_checks],
        criterion_selector=CriterionSelector(feature_names=checks),
        method=None,
        feature_extractor=FeatureExtractor(),
    )
    return DistributionPipeline(components)._analyze_sample(values, requested_checks)


def build_pipeline(
    *, model_path: str | Path, additional_statistics: Iterable[str] | None = None
) -> DistributionPipeline:
    """Build a reusable analyzer without printing or writing reports.

    Load only a trusted joblib model and calculate its selected raw statistics.
    """
    if isinstance(additional_statistics, str):
        raise ValueError("additional_statistics must be a collection of feature names")
    requested_statistics = set(additional_statistics or ())
    if any(not isinstance(name, str) or not name.strip() for name in requested_statistics):
        raise ValueError("additional_statistics must contain non-empty feature names")
    available_statistics = {spec.feature_name for spec in CRITERIA_SPECS}
    unknown_statistics = requested_statistics.difference(available_statistics)
    if unknown_statistics:
        raise ValueError("Unknown catalog statistics: " + ", ".join(sorted(unknown_statistics)))
    distributions = create_all_distributions()
    strategy = MLStrategy(model_path=model_path)
    selected_features = set(getattr(strategy, "required_features", ())) | requested_statistics
    components = PipelineComponents(
        distributions=distributions,
        criterion_selector=CriterionSelector(feature_names=selected_features),
        method=strategy,
        feature_extractor=FeatureExtractor(),
    )
    return DistributionPipeline(components)


def recommend_ml(
    data,
    *,
    model_path: str | Path,
    procedures=None,
    allow_calibration: bool = False,
    additional_statistics: Iterable[str] | None = None,
    n_bootstraps: int = 0,
    random_state: int | None = None,
):
    """Return a canonical ML recommendation with its selected catalog evidence.

    The trusted model and adjacent manifest are validated by ``build_pipeline``
    before sample analysis begins.
    """
    pipeline = build_pipeline(model_path=model_path, additional_statistics=additional_statistics)
    return pipeline.recommend(
        data,
        procedures=dict(procedures or {}),
        allow_calibration=allow_calibration,
        analysis_provider=pipeline.analyze_catalog,
        n_bootstraps=n_bootstraps,
        random_state=random_state,
    )


def evaluate_sample(
    data,
    model_path,
    save_plot=False,
    plot_path="distribution_report.png",
    save_pdf=False,
    pdf_path="distribution_report.pdf",
    n_bootstraps=0,
    random_state=42,
    pdf_mode="summary",
):
    """Recommend from a sample and print/save views of the stored result.

    Args:
        data (np.ndarray): 1D sample array.
        model_path (str | Path): Path to a trusted .joblib model with an adjacent
            compatibility manifest.
        save_plot (bool): Whether to generate 4-panel PNG chart.
        plot_path (str): Output path for the PNG plot.
        save_pdf (bool): Whether to generate a paginated recommendation PDF report.
        pdf_path (str): Output path for the PDF report.
        n_bootstraps (int): Optional stability resamples (default 0), without changing ranking.
        random_state (int | None): Seed for reproducible bootstrap resampling.
        pdf_mode (str): 'summary' (default) or 'full' with selected feature inputs.

    Returns:
        RecommendationResult: Canonical recommendation and optional diagnostics.
    """
    if pdf_mode not in {"summary", "full"}:
        raise ValueError("PDF mode must be 'summary' or 'full'")
    result = recommend_ml(
        data,
        model_path=model_path,
        n_bootstraps=n_bootstraps,
        random_state=random_state,
    )
    options = ReportOptions(detail=pdf_mode)
    print(render_result_report(result, options))

    if save_plot:
        plot_file = generate_plot_report(result, output_path=plot_path, options=options)
        print(f"🖼️ High-resolution 4-panel plot saved to: '{plot_file}'\n")

    if save_pdf:
        pdf_file = write_result_pdf(result, path=pdf_path, options=options)
        print(f"📄 PDF report saved to: '{pdf_file}'\n")

    return result
