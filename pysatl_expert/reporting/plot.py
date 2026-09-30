"""PNG presentation of an analysis result."""

import logging
from pathlib import Path

import numpy as np
import scipy.stats as stats

from pysatl_expert.models.report import Report
from pysatl_expert.reporting.common import get_scipy_dist as _get_scipy_dist


logger = logging.getLogger(__name__)


def generate_plot_report(
    data: np.ndarray,
    report: Report,
    output_path: str | Path = "distribution_report.png",
) -> Path:
    """Generate a high-resolution 4-panel visualization chart."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Plot generation requires the optional matplotlib and seaborn packages"
        ) from exc

    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
        }
    )

    output_path = Path(output_path)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=300)
    if report.bootstrap_stability is not None and report.model_confidence is not None:
        metrics_str = (
            f"Confidence: {report.model_confidence * 100:.1f}%, "
            f"Stability: {report.bootstrap_stability * 100:.1f}%"
        )
    else:
        confidence_label = (
            "Bootstrap stability"
            if report.confidence_kind == "bootstrap_stability"
            else "Model confidence"
        )
        metrics_str = f"{confidence_label}: {report.confidence * 100:.1f}%"
    title_str = f"pysatl-expert Identification: {report.distribution_name} ({metrics_str})"
    fig.suptitle(title_str, fontsize=15, fontweight="bold", y=0.98)

    data_sorted = np.sort(data)
    scipy_dist = _get_scipy_dist(report.distribution_name, report.parameters or {})

    # --- Panel 1: Empirical Histogram vs Fitted PDF ---
    ax1 = axes[0, 0]
    ax1.set_title("1. Empirical Histogram & Fitted PDF Curve")
    sns.histplot(
        data,
        kde=False,
        stat="density",
        ax=ax1,
        color="#bfdbfe",
        edgecolor="#1d4ed8",
        alpha=0.55,
        label="Sample Data",
    )

    if scipy_dist is not None:
        x_grid = np.linspace(np.min(data), np.max(data), 500)
        try:
            pdf_vals = scipy_dist.pdf(x_grid)
            ax1.plot(
                x_grid,
                pdf_vals,
                color="#b91c1c",
                lw=2.2,
                label=f"Fitted {report.distribution_name} PDF",
            )
        except Exception as exc:
            logger.warning("Could not compute the fitted PDF: %s", exc)
            ax1.text(0.5, 0.5, "Fitted PDF unavailable", ha="center", va="center")
    else:
        ax1.text(0.5, 0.5, "Fitted PDF unavailable", ha="center", va="center")
    ax1.set_xlabel("Value (x)")
    ax1.set_ylabel("Probability Density")
    ax1.legend(loc="best")

    # --- Panel 2: ECDF vs Theoretical CDF ---
    ax2 = axes[0, 1]
    ax2.set_title("2. Empirical CDF vs Theoretical CDF")
    ecdf_y = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
    ax2.plot(data_sorted, ecdf_y, color="#1d4ed8", lw=2.0, label="Empirical CDF (ECDF)")

    if scipy_dist is not None:
        try:
            cdf_vals = scipy_dist.cdf(data_sorted)
            ax2.plot(
                data_sorted,
                cdf_vals,
                color="#b91c1c",
                linestyle="--",
                lw=2.0,
                label=f"Theoretical {report.distribution_name} CDF",
            )
        except Exception as exc:
            logger.warning("Could not compute the fitted CDF: %s", exc)
            ax2.text(0.5, 0.5, "Fitted CDF unavailable", ha="center", va="center")
    else:
        ax2.text(0.5, 0.5, "Fitted CDF unavailable", ha="center", va="center")
    ax2.set_xlabel("Value (x)")
    ax2.set_ylabel("Cumulative Probability")
    ax2.legend(loc="best")

    # --- Panel 3: Q-Q Plot ---
    ax3 = axes[1, 0]
    ax3.set_title(f"3. Quantile-Quantile (Q-Q) Plot vs {report.distribution_name}")
    if scipy_dist is not None:
        try:
            (osm, osr), (slope, intercept, r) = stats.probplot(data, dist=scipy_dist, plot=None)
            ax3.scatter(
                osm,
                osr,
                color="#1d4ed8",
                alpha=0.55,
                edgecolors="none",
                s=20,
                label="Sample Quantiles",
            )
            line_x = np.array([np.min(osm), np.max(osm)])
            ax3.plot(
                line_x,
                slope * line_x + intercept,
                color="#b91c1c",
                linestyle="--",
                lw=2.0,
                label=f"Reference Line ($R^2={r**2:.3f}$)",
            )
            ax3.set_xlabel("Theoretical Quantiles")
            ax3.set_ylabel("Sample Quantiles")
            ax3.legend(loc="best")
        except Exception as exc:
            logger.warning("Could not compute the Q-Q diagnostics: %s", exc)
            ax3.text(
                0.5, 0.5, "Q-Q Plot unavailable for this parameterization", ha="center", va="center"
            )
    else:
        ax3.text(0.5, 0.5, "Q-Q Plot unavailable", ha="center", va="center")

    # --- Panel 4: Candidate Probabilities Bar Chart ---
    ax4 = axes[1, 1]
    is_bootstrap = report.confidence_kind == "bootstrap_stability"
    ax4.set_title(
        "4. ML Decision Ranking (Bootstrap Votes)"
        if is_bootstrap
        else "4. Candidate Distributions ML Probability Ranking"
    )

    if report.final_ranks:
        sorted_ranks = sorted(report.final_ranks.items(), key=lambda x: x[1])
        dists = [x[0] for x in sorted_ranks]
        probs = [x[1] * 100 for x in sorted_ranks]

        bar_colors = ["#1d4ed8" if d == report.distribution_name else "#cbd5e1" for d in dists]
        bars = ax4.barh(dists, probs, color=bar_colors, edgecolor="#94a3b8", lw=0.6, alpha=0.9)
        xlabel = "Bootstrap Vote Share (%)" if is_bootstrap else "Probability Confidence (%)"
        ax4.set_xlabel(xlabel)
        ax4.set_xlim(0, 105)

        for bar in bars:
            width = bar.get_width()
            ax4.text(
                width + 1.5,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.1f}%",
                va="center",
                fontsize=9,
            )

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    logger.info(f"Visual report saved to '{output_path}'")
    return output_path
