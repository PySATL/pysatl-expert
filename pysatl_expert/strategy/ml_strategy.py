"""ML-based strategy module for distribution classification using Random Forest."""

import logging
import sqlite3
import struct
from pathlib import Path
from typing import Dict, List

import numpy as np
import zstandard as zstd
from joblib import load as load_model

from pysatl_expert.core.strategy import AbstractStrategy
from pysatl_expert.models.feature_vector import FeatureVector
from pysatl_expert.models.report import Report


logger = logging.getLogger(__name__)


class CVCache:
    """Thread-safe cache for critical values queried from SQLite database.

    Attributes:
        db_path (str): Path to SQLite database containing limit_distributions table.
    """

    def __init__(self, db_path: str):
        """Initialize the cache and preload all critical values from the database.

        Args:
            db_path (str): Path to SQLite database file.

        Raises:
            FileNotFoundError: If the database file does not exist.
            sqlite3.Error: If the database query fails.
        """
        self.db_path = db_path
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load all critical values from the database into memory and build interpolation arrays."""
        if not Path(self.db_path).exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        raw_dict: Dict[str, Dict[int, float]] = {}

        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT DISTINCT criterion_code, sample_size, results_statistics
                FROM limit_distributions
                ORDER BY criterion_code, sample_size
                """
            )

            rows = cursor.fetchall()
            logger.info(f"Loading {len(rows)} critical value records from database...")

            for row in rows:
                criterion_code = row["criterion_code"]
                sample_size = row["sample_size"]
                results_blob = row["results_statistics"]

                try:
                    results_array = self._decompress_results(results_blob)
                    cv = float(np.percentile(results_array, 95))

                    code_lower = criterion_code.lower()
                    if code_lower not in raw_dict:
                        raw_dict[code_lower] = {}

                    raw_dict[code_lower][sample_size] = cv

                except Exception as e:
                    logger.warning(
                        f"Error processing CV for {criterion_code} (n={sample_size}): {e}"
                    )

            conn.close()

            for code, size_map in raw_dict.items():
                sorted_sizes = sorted(size_map.keys())
                sorted_cvs = [size_map[s] for s in sorted_sizes]
                self._cache[code] = {
                    "sizes": np.array(sorted_sizes, dtype=np.float64),
                    "cvs": np.array(sorted_cvs, dtype=np.float64),
                }

            logger.info(f"CVCache loaded with {len(self._cache)} criterion codes")

        except sqlite3.Error as e:
            logger.error(f"Database error while loading CV cache: {e}")
            raise

    @staticmethod
    def _decompress_results(compressed_data: bytes) -> np.ndarray:
        """Decompress zstd-compressed results_statistics blob.

        Format: [6-byte header][zstd-compressed 32-bit floats]

        Args:
            compressed_data: Raw blob from database.

        Returns:
            Numpy array of float64 values.
        """
        if len(compressed_data) < 6:
            raise ValueError(f"Invalid data: expected at least 6 bytes, got {len(compressed_data)}")

        zstd_data = compressed_data[6:]
        dctx = zstd.ZstdDecompressor()
        decompressed = dctx.decompress(zstd_data)

        float_size = 4
        num_floats = len(decompressed) // float_size

        if num_floats == 0:
            return np.array([], dtype=np.float64)

        data = struct.unpack(f"{num_floats}f", decompressed[: num_floats * float_size])
        return np.array(data, dtype=np.float64)

    def get_cv(self, criterion_code: str, sample_size: int) -> float | None:
        """Retrieve the linearly interpolated critical value for a given test and sample size.

        Args:
            criterion_code: Name of the statistical test (e.g., "KS_NORMALITY_GOODNESS_OF_FIT")
            sample_size: Sample size used in Monte Carlo simulation.

        Returns:
            Linearly interpolated 95th percentile critical value, or None if criterion not found.
        """
        code_lower = criterion_code.lower()

        if code_lower not in self._cache:
            logger.debug(f"Criterion code not in cache: {criterion_code}")
            return None

        entry = self._cache[code_lower]
        sizes = entry["sizes"]
        cvs = entry["cvs"]

        if len(sizes) == 0:
            return None

        return float(np.interp(sample_size, sizes, cvs))


class MLStrategy(AbstractStrategy):
    """Machine Learning-based strategy using a pre-trained Random Forest classifier.

    This strategy binarizes raw GoF test results using critical values from Monte
    Carlo simulations, then uses the trained RF model to predict the most likely
    distribution with confidence scores.

    Attributes:
        model: Loaded joblib Random Forest model.
        cv_cache: Critical value cache for binarization.
        _class_names: Ordered list of distribution class names expected by the model.
    """

    def __init__(self, model_path: str, cv_database_path: str):
        """Initialize the ML strategy with a trained model and critical value database.

        Args:
            model_path: Path to the trained Random Forest model (joblib format).
            cv_database_path: Path to the SQLite database with critical values.

        Raises:
            FileNotFoundError: If either file does not exist.
            Exception: If model loading fails.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            self.model = load_model(model_path)
            logger.info(f"Loaded Random Forest model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        self._class_names = sorted(self.model.classes_.tolist())
        logger.info(f"Model classes: {self._class_names}")

        self.cv_cache = CVCache(cv_database_path)

    def _binarize_vector(self, raw_flat_vector: List[float], sample_size: int) -> np.ndarray:
        """Binarize raw continuous GoF test results using critical values.

        The vector layout is:
        - First 6 elements: Continuous statistics (sample_size, skew, kurtosis, etc.)
        - Remaining elements: GoF test results (to be binarized)

        Binarization logic:
        - If value <= critical_value: set to 1.0 (hypothesis accepted)
        - If value > critical_value: set to 0.0 (hypothesis rejected)
        - If value == -1.0: keep as -1.0 (inapplicable)
        - Continuous statistics: leave unchanged

        Args:
            raw_flat_vector: Raw feature vector from pipeline (continuous values).
            sample_size: Size of the original sample (used for CV lookup).

        Returns:
            Binarized numpy array ready for RF model prediction.
        """
        binarized = np.array(raw_flat_vector, dtype=np.float64)

        stat_keys = FeatureVector.STAT_KEYS
        num_stats = len(stat_keys)

        for idx, (dist_name, crit_code) in enumerate(FeatureVector.CRITERIA_SCHEMA):
            vector_idx = num_stats + idx

            if vector_idx >= len(binarized):
                logger.warning(f"Vector index {vector_idx} out of bounds")
                break

            raw_value = binarized[vector_idx]

            if raw_value == -1.0:
                continue

            criterion_code_upper = crit_code.upper()
            cv = self.cv_cache.get_cv(criterion_code_upper, sample_size)

            if cv is None:
                logger.debug(
                    f"No critical value for {criterion_code_upper} (n={sample_size}), "
                    f"keeping value as-is"
                )
                continue

            if raw_value <= cv:
                binarized[vector_idx] = 1.0
            else:
                binarized[vector_idx] = 0.0

        return binarized

    def predict_report(
        self, base_fv: FeatureVector, bootstrap_fvs: List[FeatureVector] | None = None
    ) -> Report:
        """Generate the final identification report using the Random Forest model.

        Process:
        1. Extract raw flat vector from base_fv.
        2. Determine sample size from base_fv.sample_stats.
        3. Binarize the vector using critical values.
        4. Feed to RF model for prediction.
        5. Extract top winner and probabilities.
        6. Return Report with confidence from RF probabilities.

        Args:
            base_fv: FeatureVector calculated on the original sample.
            bootstrap_fvs: Optional list of FeatureVectors from bootstrap resampling.
                If provided, used for confidence calculation via voting.

        Returns:
            Report object with predicted distribution, confidence, and scores.
        """
        raw_vector = base_fv.as_flat_list(missing_value=-1.0)

        sample_size = int(base_fv.sample_stats.get("sample_size", 100))

        binarized_vector = self._binarize_vector(raw_vector, sample_size)

        X = binarized_vector.reshape(1, -1)

        probabilities = self.model.predict_proba(X)[0]

        winner_idx = np.argmax(probabilities)
        winner = self._class_names[winner_idx]
        confidence = float(probabilities[winner_idx])

        all_scores = {name: float(prob) for name, prob in zip(self._class_names, probabilities)}

        if bootstrap_fvs:
            votes = []
            for fv in bootstrap_fvs:
                boot_vector = fv.as_flat_list(missing_value=-1.0)
                boot_sample_size = int(fv.sample_stats.get("sample_size", 100))
                binarized_boot = self._binarize_vector(boot_vector, boot_sample_size)
                X_boot = binarized_boot.reshape(1, -1)
                boot_pred = self.model.predict(X_boot)[0]
                votes.append(boot_pred)

            unique, counts = np.unique(votes, return_counts=True)
            best_idx = np.argmax(counts)
            winner = unique[best_idx]
            confidence = float(counts[best_idx]) / len(votes)

        return Report(
            distribution_name=winner,
            confidence=round(confidence, 3),
            all_scores=all_scores,
            final_ranks=all_scores,
        )
