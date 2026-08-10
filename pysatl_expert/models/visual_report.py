import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns

from pysatl_expert.models.report import Report


logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
})


def _get_scipy_dist(dist_name: str, params: dict):
    """Map pysatl distribution name and parameter dictionary to scipy.stats distribution."""
    name_lower = dist_name.lower()

    if "normal" in name_lower and "log" not in name_lower:
        loc = params.get("mu", params.get("loc", 0.0))
        scale = params.get("std", params.get("scale", 1.0))
        return stats.norm(loc=loc, scale=scale)

    elif "lognormal" in name_lower or "log_normal" in name_lower:
        s = params.get("s", params.get("shape", 1.0))
        loc = params.get("loc", 0.0)
        scale = params.get("scale", 1.0)
        return stats.lognorm(s=s, loc=loc, scale=scale)

    elif "exponential" in name_lower:
        loc = params.get("loc", 0.0)
        scale = params.get("scale", 1.0)
        return stats.expon(loc=loc, scale=scale)

    elif "uniform" in name_lower:
        if "a" in params and "b" in params:
            loc = params["a"]
            scale = params["b"] - params["a"]
        else:
            loc = params.get("loc", 0.0)
            scale = params.get("scale", 1.0)
        return stats.uniform(loc=loc, scale=scale)

    elif "student" in name_lower:
        df = params.get("df", 5.0)
        loc = params.get("loc", 0.0)
        scale = params.get("scale", 1.0)
        return stats.t(df=df, loc=loc, scale=scale)

    elif "gamma" in name_lower:
        a = params.get("shape", params.get("a", 2.0))
        loc = params.get("loc", 0.0)
        scale = params.get("scale", 1.0)
        return stats.gamma(a=a, loc=loc, scale=scale)

    elif "weibull" in name_lower:
        c = params.get("shape", params.get("c", 1.5))
        loc = params.get("loc", 0.0)
        scale = params.get("scale", 1.0)
        return stats.weibull_min(c=c, loc=loc, scale=scale)

    elif "beta" in name_lower:
        a = params.get("a", 2.0)
        b = params.get("b", 2.0)
        loc = params.get("loc", 0.0)
        scale = params.get("scale", 1.0)
        return stats.beta(a=a, b=b, loc=loc, scale=scale)

    return None


def generate_text_report(data: np.ndarray, report: Report) -> str:
    """Generate a clean human-readable ASCII text report."""
    n_samples = len(data)
    d_min, d_max = np.min(data), np.max(data)
    mean_val, std_val = np.mean(data), np.std(data)
    skew_val = float(stats.skew(data))
    kurt_val = float(stats.kurtosis(data))

    lines = []
    lines.append("=" * 80)
    lines.append("             pysatl-expert: Distribution Identification Report             ")
    lines.append("=" * 80)
    lines.append("📊 Sample Characteristics:")
    lines.append(f"  • Sample Size (N):    {n_samples}")
    lines.append(f"  • Domain Range:       [{d_min:.4f}, {d_max:.4f}]")
    lines.append(f"  • Mean & Std Dev:     Mean = {mean_val:.4f} | Std = {std_val:.4f}")
    lines.append(f"  • Skewness & Kurt:    Skew = {skew_val:.4f} | Kurtosis = {kurt_val:.4f}")
    lines.append("-" * 80)

    lines.append(f"🏆 IDENTIFIED BEST FIT: {report.distribution_name.upper()}")
    lines.append(f"  • ML Expert Confidence: {report.confidence * 100:.2f}%")

    if report.parameters:
        param_str = ", ".join(f"{k} = {v:.4f}" if isinstance(v, (float, int)) else f"{k} = {v}" for k, v in report.parameters.items())
        lines.append(f"  • Estimated Parameters: {param_str}")
    lines.append("-" * 80)

    if report.final_ranks:
        lines.append("📊 Candidate Distributions Probability Ranking:")
        sorted_ranks = sorted(report.final_ranks.items(), key=lambda x: x[1], reverse=True)
        max_len = max(len(k) for k in report.final_ranks.keys())

        for dist_k, prob_v in sorted_ranks:
            pct = prob_v * 100
            bar_len = int(prob_v * 40)
            bar = "█" * bar_len
            lines.append(f"  {dist_k:<{max_len}} : {pct:6.2f}% {bar}")
        lines.append("-" * 80)

    if report.all_scores:
        winner_scores = report.all_scores.get(report.distribution_name, {})
        if isinstance(winner_scores, dict):
            passed_crit = [k for k, v in winner_scores.items() if v == 1.0 or v is True]
            failed_crit = [k for k, v in winner_scores.items() if v == 0.0 or v is False]
            total_crit = len(winner_scores)
        else:
            passed_crit = [k for k, v in report.all_scores.items() if isinstance(v, (int, float, bool)) and v == 1.0]
            failed_crit = [k for k, v in report.all_scores.items() if isinstance(v, (int, float, bool)) and v == 0.0]
            total_crit = len(report.all_scores)

        if total_crit > 0:
            lines.append("💡 Goodness-of-Fit (GoF) Criteria Evaluation for Winner:")
            lines.append(f"  • Total Criteria Evaluated: {total_crit}")
            lines.append(f"  • Passed Criteria (H0 Accepted at α=0.05): {len(passed_crit)} / {total_crit}")
            if passed_crit:
                top_passed = ", ".join(passed_crit[:10])
                if len(passed_crit) > 10:
                    top_passed += f" ... (+{len(passed_crit) - 10} more)"
                lines.append(f"    ✅ Passed: {top_passed}")
            if failed_crit:
                top_failed = ", ".join(failed_crit[:6])
                if len(failed_crit) > 6:
                    top_failed += f" ... (+{len(failed_crit) - 6} more)"
                lines.append(f"    ❌ Rejected: {top_failed}")

    lines.append("=" * 80)
    return "\n".join(lines)


def generate_plot_report(
    data: np.ndarray,
    report: Report,
    output_path: str | Path = "distribution_report.png",
) -> Path:
    """Generate a high-resolution 4-panel visualization chart."""
    output_path = Path(output_path)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=300)
    fig.suptitle(
        f"pysatl-expert Identification: {report.distribution_name} (Confidence: {report.confidence * 100:.1f}%)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

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
        color="#3498db",
        edgecolor="black",
        alpha=0.6,
        label="Sample Data",
    )

    if scipy_dist is not None:
        x_grid = np.linspace(np.min(data), np.max(data), 500)
        try:
            pdf_vals = scipy_dist.pdf(x_grid)
            ax1.plot(x_grid, pdf_vals, "r-", lw=2.5, label=f"Fitted {report.distribution_name} PDF")
        except Exception:
            pass
    ax1.set_xlabel("Value (x)")
    ax1.set_ylabel("Probability Density")
    ax1.legend(loc="best")

    # --- Panel 2: ECDF vs Theoretical CDF ---
    ax2 = axes[0, 1]
    ax2.set_title("2. Empirical CDF vs Theoretical CDF")
    ecdf_y = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
    ax2.plot(data_sorted, ecdf_y, "b-", lw=2, label="Empirical CDF (ECDF)")

    if scipy_dist is not None:
        try:
            cdf_vals = scipy_dist.cdf(data_sorted)
            ax2.plot(data_sorted, cdf_vals, "r--", lw=2, label=f"Theoretical {report.distribution_name} CDF")
        except Exception:
            pass
    ax2.set_xlabel("Value (x)")
    ax2.set_ylabel("Cumulative Probability")
    ax2.legend(loc="best")

    # --- Panel 3: Q-Q Plot ---
    ax3 = axes[1, 0]
    ax3.set_title(f"3. Quantile-Quantile (Q-Q) Plot vs {report.distribution_name}")
    if scipy_dist is not None:
        try:
            (osm, osr), (slope, intercept, r) = stats.probplot(data, dist=scipy_dist, plot=None)
            ax3.scatter(osm, osr, color="#2ecc71", alpha=0.7, edgecolors="none", s=25, label="Sample Quantiles")
            line_x = np.array([np.min(osm), np.max(osm)])
            ax3.plot(line_x, slope * line_x + intercept, "r--", lw=2, label=f"Reference Line ($R^2={r**2:.3f}$)")
            ax3.set_xlabel("Theoretical Quantiles")
            ax3.set_ylabel("Sample Quantiles")
            ax3.legend(loc="best")
        except Exception:
            ax3.text(0.5, 0.5, "Q-Q Plot unavailable for this parameterization", ha="center", va="center")
    else:
        ax3.text(0.5, 0.5, "Q-Q Plot unavailable", ha="center", va="center")

    # --- Panel 4: Candidate Probabilities Bar Chart ---
    ax4 = axes[1, 1]
    ax4.set_title("4. Candidate Distributions ML Probability Ranking")

    if report.final_ranks:
        sorted_ranks = sorted(report.final_ranks.items(), key=lambda x: x[1])
        dists = [x[0] for x in sorted_ranks]
        probs = [x[1] * 100 for x in sorted_ranks]

        colors = ["#2ecc71" if d == report.distribution_name else "#95a5a6" for d in dists]
        bars = ax4.barh(dists, probs, color=colors, edgecolor="black", alpha=0.85)
        ax4.set_xlabel("Probability Confidence (%)")
        ax4.set_xlim(0, 105)

        for bar in bars:
            width = bar.get_width()
            ax4.text(width + 1.5, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", va="center", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    logger.info(f"Visual report saved to '{output_path}'")
    return output_path
