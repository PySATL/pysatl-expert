import numpy as np
import scipy.stats as stats


class FeatureExtractor:
    """Calculate descriptive statistics used by the model and reports."""

    def calculate_sample_stats(self, data: np.ndarray) -> dict[str, float | int]:
        """Compute descriptive sample statistics.

        Args:
            data (np.ndarray): Raw numerical sample array to profile.

        Returns:
            dict[str, float | int]: Statistics for model features and presentation.
        """
        data_min = float(np.min(data))
        data_max = float(np.max(data))
        n = len(data)
        mean = float(np.mean(data))
        median = float(np.median(data))

        skew = stats.skew(data)
        kurt = stats.kurtosis(data)

        std_val = float(np.std(data))
        variance = float(np.var(data))

        q25, q75 = np.percentile(data, [25, 75])
        iqr = q75 - q25
        relative_iqr = iqr / std_val if std_val > 0.0 else 0.0

        entropy = stats.entropy(np.histogram(data, bins="auto")[0])

        return {
            "min": data_min,
            "max": data_max,
            "sample_size": int(n),
            "mean": mean,
            "median": median,
            "standard_deviation": std_val,
            "variance": variance,
            "skew": float(skew),
            "kurtosis": float(kurt),
            "iqr": float(iqr),
            "relative_iqr": float(relative_iqr),
            "entropy": float(entropy),
        }
