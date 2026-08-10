import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

repo_root = str(Path(__file__).parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import pandas as pd
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
from pysatl_expert.models.feature_vector import FeatureVector
from pysatl_expert.pipeline import DistributionPipeline
from pysatl_expert.strategy.ml_strategy import MLStrategy


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_pipeline = None
_strategy = None


def _init_worker(db_path: str):
    """Initialize pipeline components once per worker process."""
    global _pipeline, _strategy

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

    dummy_model_path = Path(repo_root) / "pysatl_expert" / "rf_expert_model.joblib"
    if not dummy_model_path.exists():
        import joblib
        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(n_estimators=2, random_state=42)
        X_dummy = np.zeros(
            (8, len(FeatureVector.STAT_KEYS) + len(FeatureVector.CRITERIA_SCHEMA))
        )
        y_dummy = np.array(
            [
                "Normal",
                "Exponential",
                "Weibull",
                "Uniform",
                "Student",
                "Gamma",
                "Beta",
                "LogNormal",
            ]
        )
        rf.fit(X_dummy, y_dummy)
        joblib.dump(rf, dummy_model_path)

    _strategy = MLStrategy(model_path=str(dummy_model_path), cv_database_path=db_path)

    components = PipelineComponents(
        distributions=distributions,
        criterion_selector=DynamicCriterionSelector(),
        strategy=_strategy,
        feature_extractor=FeatureExtractor(),
    )

    _pipeline = DistributionPipeline(components)


def generate_single_sample(dist_name: str) -> tuple[np.ndarray, str]:
    """Generate a single random sample for a target distribution with randomized parameters."""
    N = int(np.random.randint(30, 1001))

    if dist_name == "Normal":
        loc = float(np.random.uniform(-10, 10))
        scale = float(np.random.uniform(0.5, 5.0))
        sample = st.norm.rvs(loc=loc, scale=scale, size=N)

    elif dist_name == "Exponential":
        loc = float(np.random.uniform(-10, 10))
        scale = float(np.random.uniform(0.5, 5.0))
        sample = st.expon.rvs(loc=loc, scale=scale, size=N)

    elif dist_name == "Weibull":
        c = float(np.random.uniform(0.5, 3.0))
        loc = float(np.random.uniform(-5, 5))
        scale = float(np.random.uniform(0.5, 5.0))
        sample = st.weibull_min.rvs(c=c, loc=loc, scale=scale, size=N)

    elif dist_name == "Uniform":
        a = float(np.random.uniform(-10, 5))
        b = a + float(np.random.uniform(1.0, 20.0))
        sample = st.uniform.rvs(loc=a, scale=b - a, size=N)

    elif dist_name == "Student":
        df = float(np.random.uniform(1.5, 30.0))
        loc = float(np.random.uniform(-5, 5))
        scale = float(np.random.uniform(0.5, 5.0))
        sample = st.t.rvs(df=df, loc=loc, scale=scale, size=N)

    elif dist_name == "Gamma":
        a = float(np.random.uniform(0.5, 5.0))
        loc = float(np.random.uniform(-5, 5))
        scale = float(np.random.uniform(0.5, 5.0))
        sample = st.gamma.rvs(a=a, loc=loc, scale=scale, size=N)

    elif dist_name == "Beta":
        a = float(np.random.uniform(0.5, 5.0))
        b = float(np.random.uniform(0.5, 5.0))
        sample = st.beta.rvs(a=a, b=b, size=N)

    elif dist_name == "LogNormal":
        s = float(np.random.uniform(0.1, 1.5))
        loc = float(np.random.uniform(-5, 5))
        scale = float(np.random.uniform(0.5, 5.0))
        sample = st.lognorm.rvs(s=s, loc=loc, scale=scale, size=N)

    else:
        raise ValueError(f"Unknown target distribution: {dist_name}")

    return sample, dist_name


def _process_item(dist_name: str) -> list[float | str] | None:
    """Worker task: Generate, evaluate and binarize a sample vector."""
    try:
        sample, target_label = generate_single_sample(dist_name)
        base_fv, _ = _pipeline._evaluate_sample(sample)
        sample_size = int(base_fv.sample_stats.get("sample_size", len(sample)))
        raw_vector = base_fv.as_flat_list(missing_value=-1.0)
        binarized_vector = _strategy._binarize_vector(raw_vector, sample_size)
        row = list(binarized_vector) + [target_label]
        return row
    except Exception as e:
        logger.warning(f"Error generating sample for {dist_name}: {e}")
        return None


def main():
    target_distributions = [
        "Normal",
        "Exponential",
        "Weibull",
        "Uniform",
        "Student",
        "Gamma",
        "Beta",
        "LogNormal",
    ]
    samples_per_dist = 2000
    total_samples = len(target_distributions) * samples_per_dist

    db_path = str(Path(repo_root) / "pysatl_expert" / "expert_cv_database.sqlite")
    output_csv = Path(repo_root) / "pysatl_expert" / "expert_ml_dataset_binary.csv"

    logger.info(f"Generating ML Dataset ({total_samples} samples)...")
    logger.info(f"Database path: {db_path}")
    logger.info(f"Output CSV path: {output_csv}")

    tasks = []
    for dist in target_distributions:
        tasks.extend([dist] * samples_per_dist)

    np.random.shuffle(tasks)

    n_workers = min(6, os.cpu_count() or 4)
    logger.info(f"Launching {n_workers} parallel workers...")

    t0 = time.time()
    results = []
    done_count = 0

    with mp.Pool(processes=n_workers, initializer=_init_worker, initargs=(db_path,)) as pool:
        for row in pool.imap_unordered(_process_item, tasks, chunksize=10):
            if row is not None:
                results.append(row)
            done_count += 1
            if done_count % 1000 == 0 or done_count == total_samples:
                elapsed = time.time() - t0
                speed = done_count / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Progress: {done_count}/{total_samples} samples ({done_count/total_samples*100:.1f}%) | Speed: {speed:.1f} samples/sec"
                )

    elapsed_total = time.time() - t0
    logger.info(f"Generation completed in {elapsed_total:.1f} sec ({elapsed_total/60:.2f} min).")

    feature_cols = list(FeatureVector.STAT_KEYS)
    for dist_name, crit_code in FeatureVector.CRITERIA_SCHEMA:
        feature_cols.append(f"{crit_code.upper()}")

    columns = feature_cols + ["Target"]
    df = pd.DataFrame(results, columns=columns)

    logger.info(f"Generated Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns.")
    df.to_csv(output_csv, index=False)
    logger.info(f"Successfully saved dataset to '{output_csv}'")


if __name__ == "__main__":
    main()
