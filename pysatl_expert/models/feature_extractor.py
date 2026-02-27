import numpy as np
import scipy.stats as stats


class FeatureExtractor:
    """
    Service for calculating intrinsic statistical properties of a data sample.

    Computes a set of descriptive metrics used to build a profile of the input
    data. These features characterize the shape and complexity of the sample,
    independent of its scale, providing the necessary inputs for decision-making
    strategies.
    """

    def __init__(self):
        """
        Initializes the feature extraction service.
        """
        pass

    def calculate_sample_stats(self, data: np.ndarray) -> dict:
        """
        Computes a dictionary of scale-invariant and robust sample statistics.

        The extracted features include:
        - Boundary values (min, max) for domain validation.
        - Classical shape moments (skewness, kurtosis).
        - Dispersion metrics (variation, relative IQR).
        - Complexity measures (entropy).

        Args:
            data (np.ndarray): The raw numerical sample to profile.

        Returns:
            dict: A collection of calculated features (floats and ints).
        """
        data_min = np.min(data)
        data_max = np.max(data)
        n = len(data)

        skew = stats.skew(data)
        kurt = stats.kurtosis(data)

        mean_val = float(np.mean(data))
        std_val = float(np.std(data))
        variation = std_val / mean_val if abs(mean_val) > 1e-9 else 0

        q25, q50, q75 = np.percentile(data, [25, 50, 75])
        iqr = q75 - q25
        relative_iqr = iqr / q50 if q50 != 0 else 0

        entropy = stats.entropy(np.histogram(data, bins="auto")[0])

        return {
            "min": float(data_min),
            "max": float(data_max),
            "sample_size": int(n),
            "skew": float(skew),
            "kurtosis": float(kurt),
            "coef_of_variation": float(variation),
            "relative_iqr": float(relative_iqr),
            "entropy": float(entropy),
        }
