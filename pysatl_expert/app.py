from pathlib import Path

from pysatl_expert.core.pipeline_components import PipelineComponents
from pysatl_expert.criteria.selectors.selector import CriterionSelector
from pysatl_expert.distributions.beta import BetaDistribution
from pysatl_expert.distributions.exponential import ExponentialDistribution
from pysatl_expert.distributions.gamma import GammaDistribution
from pysatl_expert.distributions.log_normal import LogNormalDistribution
from pysatl_expert.distributions.normal import NormalDistribution
from pysatl_expert.distributions.student import StudentDistribution
from pysatl_expert.distributions.uniform import UniformDistribution
from pysatl_expert.distributions.weibull import WeibullDistribution
from pysatl_expert.models.feature_extractor import FeatureExtractor
from pysatl_expert.pipeline import DistributionPipeline
from pysatl_expert.reporting.pdf import generate_pdf_report
from pysatl_expert.reporting.plot import generate_plot_report
from pysatl_expert.reporting.text import generate_text_report
from pysatl_expert.strategy.ml_strategy import MLStrategy


def build_pipeline(*, model_path: str | Path) -> DistributionPipeline:
    """Build a reusable analyzer without printing or writing reports.

    Load only a trusted joblib model and calculate its selected raw statistics.
    """
    distributions = [
        NormalDistribution(),
        ExponentialDistribution(),
        WeibullDistribution(),
        UniformDistribution(),
        StudentDistribution(),
        GammaDistribution(),
        BetaDistribution(),
        LogNormalDistribution(),
    ]
    strategy = MLStrategy(model_path=model_path)
    components = PipelineComponents(
        distributions=distributions,
        criterion_selector=CriterionSelector(
            feature_names=getattr(strategy, "required_features", None)
        ),
        strategy=strategy,
        feature_extractor=FeatureExtractor(),
    )
    return DistributionPipeline(components)


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
    """Evaluate empirical sample data and print/save visual identification report.

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
        Report: Complete identification report.
    """
    if pdf_mode not in {"summary", "full"}:
        raise ValueError("PDF mode must be 'summary' or 'full'")
    pipeline = build_pipeline(model_path=model_path)
    identify_options = {"n_bootstraps": n_bootstraps, "random_state": random_state}
    report = pipeline.identify_best(data, **identify_options)

    text_report = generate_text_report(data, report)
    print(text_report)

    if save_plot:
        plot_file = generate_plot_report(data, report, output_path=plot_path)
        print(f"🖼️ High-resolution 4-panel plot saved to: '{plot_file}'\n")

    if save_pdf:
        pdf_file = generate_pdf_report(data, report, output_path=pdf_path, mode=pdf_mode)
        print(f"📄 Publication-quality PDF report saved to: '{pdf_file}'\n")

    return report
