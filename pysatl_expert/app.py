import sys
from pathlib import Path


expert_root = str(Path(__file__).parents[1])
if expert_root not in sys.path:
    sys.path.insert(0, expert_root)

import scipy.stats as st

from pysatl_expert.core.pipeline_components import PipelineComponents
from pysatl_expert.criteria.selectors.dynamic_selector import DynamicCriterionSelector
from pysatl_expert.distributions.beta import BetaDistribution
from pysatl_expert.distributions.exponential import ExponentialDistribution
from pysatl_expert.distributions.gamma import GammaDistribution
from pysatl_expert.distributions.log_normal import LogNormalDistribution
from pysatl_expert.distributions.normal import NormalDistribution
from pysatl_expert.distributions.student import StudentDistribution
from pysatl_expert.distributions.uniform import UniformDistribution
from pysatl_expert.distributions.weibull import WeibullDistribution
from pysatl_expert.models.feature_extractor import FeatureExtractor
from pysatl_expert.models.visual_report import generate_plot_report, generate_text_report
from pysatl_expert.pipeline import DistributionPipeline
from pysatl_expert.strategy.ml_strategy import MLStrategy


def evaluate_sample(data, model_path=None, db_path=None, save_plot=True, plot_path="distribution_report.png"):
    """Evaluate empirical sample data and print/save visual identification report.

    Args:
        data (np.ndarray): 1D sample array.
        model_path (str | None): Path to trained .joblib model.
        db_path (str | None): Path to SQLite CV database.
        save_plot (bool): Whether to generate 4-panel PNG chart.
        plot_path (str): Output path for the PNG plot.

    Returns:
        Report: Complete identification report.
    """
    expert_dir = Path(__file__).parent
    model_path = model_path or (expert_dir / "rf_expert_model.joblib")
    db_path = db_path or (expert_dir / "expert_cv_database.sqlite")

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

    strategy = MLStrategy(model_path=model_path, cv_database_path=db_path)
    components = PipelineComponents(
        distributions=distributions,
        criterion_selector=DynamicCriterionSelector(),
        strategy=strategy,
        feature_extractor=FeatureExtractor(),
    )

    pipeline = DistributionPipeline(components)
    report = pipeline.identify_best(data, n_bootstraps=0)

    text_report = generate_text_report(data, report)
    print(text_report)

    if save_plot:
        plot_file = generate_plot_report(data, report, output_path=plot_path)
        print(f"🖼️ High-resolution 4-panel plot saved to: '{plot_file}'\n")

    return report


def main():
    """Run demonstration pipeline on synthetic empirical samples."""
    print("=== Initializing pysatl-expert Machine Learning Expert System ===")

    test_samples = {
        "Sample_Normal": st.norm.rvs(loc=10.0, scale=2.5, size=450),
        "Sample_Exponential": st.expon.rvs(scale=15.0, size=300),
        "Sample_Weibull": st.weibull_min.rvs(c=1.8, scale=5.0, size=600),
    }

    for name, data in test_samples.items():
        print(f"\n>>> Evaluating empirical sample: {name} (N={len(data)})")
        plot_file = f"result_{name}.png"
        evaluate_sample(data, save_plot=True, plot_path=plot_file)


if __name__ == "__main__":
    main()
