"""Text presentation of an analysis result; no plotting dependencies."""

import numpy as np

from pysatl_expert.models.report import Report
from pysatl_expert.reporting.formatting import format_raw_statistic as _format_raw_statistic


def _format_text_gof_summary(report: Report) -> list[str]:
    """Show raw winner statistics without interpreting them as test decisions."""
    if not report.all_scores:
        return []

    winner_scores = report.all_scores.get(report.distribution_name)
    if winner_scores is None and all(
        not isinstance(value, dict) for value in report.all_scores.values()
    ):
        winner_scores = report.all_scores
    if not winner_scores or not isinstance(winner_scores, dict):
        return []

    finite_count = sum(
        value is not None and np.isfinite(float(value)) for value in winner_scores.values()
    )
    lines = [
        f"Raw GoF statistics for {report.distribution_name}:",
        f"  • Finite statistics: {finite_count} / {len(winner_scores)}",
    ]
    lines.extend(
        f"  • {name}: {_format_raw_statistic(value)}" for name, value in winner_scores.items()
    )
    return lines


def generate_text_report(data: np.ndarray, report: Report) -> str:
    """Generate a clean human-readable ASCII text report."""
    del data
    sample_statistics = report.sample_statistics
    required_statistics = {
        "sample_size",
        "min",
        "max",
        "mean",
        "standard_deviation",
        "skew",
        "kurtosis",
    }
    missing_statistics = required_statistics.difference(sample_statistics)
    if missing_statistics:
        raise ValueError(
            "Report is missing descriptive sample statistics: "
            + ", ".join(sorted(missing_statistics))
        )

    lines = []
    lines.append("=" * 80)
    lines.append("             pysatl-expert: Distribution Identification Report             ")
    lines.append("=" * 80)
    lines.append("📊 Sample Characteristics:")
    lines.append(f"  • Sample Size (N):    {sample_statistics['sample_size']}")
    lines.append(
        f"  • Domain Range:       [{sample_statistics['min']:.4f}, "
        f"{sample_statistics['max']:.4f}]"
    )
    lines.append(
        f"  • Mean & Std Dev:     Mean = {sample_statistics['mean']:.4f} | "
        f"Std = {sample_statistics['standard_deviation']:.4f}"
    )
    lines.append(
        f"  • Skewness & Kurt:    Skew = {sample_statistics['skew']:.4f} | "
        f"Kurtosis = {sample_statistics['kurtosis']:.4f}"
    )
    lines.append("-" * 80)

    lines.append(f"🏆 IDENTIFIED BEST FIT: {report.distribution_name.upper()}")
    if report.model_confidence is not None:
        lines.append(f"  • ML Model Confidence: {report.model_confidence * 100:.2f}%")
    if report.bootstrap_stability is not None:
        lines.append(f"  • Bootstrap Stability:  {report.bootstrap_stability * 100:.2f}%")
    if report.model_confidence is None and report.bootstrap_stability is None:
        confidence_label = (
            "Bootstrap Stability"
            if report.confidence_kind == "bootstrap_stability"
            else "ML Expert Confidence"
        )
        lines.append(f"  • {confidence_label}: {report.confidence * 100:.2f}%")

    if report.parameters:
        param_str = ", ".join(
            f"{k} = {v:.4f}" if isinstance(v, (float, int)) else f"{k} = {v}"
            for k, v in report.parameters.items()
        )
        lines.append(f"  • Estimated Parameters: {param_str}")
    lines.append("-" * 80)

    if report.final_ranks:
        ranking_label = (
            "Bootstrap Vote Ranking"
            if report.confidence_kind == "bootstrap_stability"
            else "Candidate Distributions Model Score Ranking"
        )
        lines.append(f"📊 {ranking_label}:")
        sorted_ranks = sorted(report.final_ranks.items(), key=lambda x: x[1], reverse=True)
        max_len = max(len(k) for k in report.final_ranks.keys())

        for dist_k, prob_v in sorted_ranks:
            pct = prob_v * 100
            bar_len = int(prob_v * 40)
            bar = "█" * bar_len
            lines.append(f"  {dist_k:<{max_len}} : {pct:6.2f}% {bar}")
        lines.append("-" * 80)

    lines.extend(_format_text_gof_summary(report))
    lines.append("=" * 80)
    return "\n".join(lines)
