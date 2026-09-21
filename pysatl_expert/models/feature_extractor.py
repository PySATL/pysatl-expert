import numpy as np
import scipy.stats as stats


class FeatureExtractor:
    """Service for calculating intrinsic statistical properties of a sample.

    Computes descriptive metrics used to profile the shape and complexity of input data.
    """

    def calculate_sample_stats(self, data: np.ndarray) -> dict[str, float | int]:
        """Compute scale-invariant and robust sample statistics.

        Args:
            data (np.ndarray): Raw numerical sample array to profile.

        Returns:
            dict[str, float | int]: Calculated feature map including min, max, sample_size,
                skew, kurtosis, relative_iqr, and entropy.
        """
        data_min = np.min(data)
        data_max = np.max(data)
        n = len(data)

        skew = stats.skew(data)
        kurt = stats.kurtosis(data)

        std_val = float(np.std(data))

        q25, q75 = np.percentile(data, [25, 75])
        iqr = q75 - q25
        relative_iqr = iqr / std_val if std_val > 0.0 else 0.0

        entropy = stats.entropy(np.histogram(data, bins="auto")[0])

        return {
            "min": float(data_min),
            "max": float(data_max),
            "sample_size": int(n),
            "skew": float(skew),
            "kurtosis": float(kurt),
            "relative_iqr": float(relative_iqr),
            "entropy": float(entropy),
        }
