import json
import logging
import sys
from pathlib import Path


repo_root = str(Path(__file__).parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from pysatl_expert.models.hierarchical_model import HierarchicalExpertModel


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Train HierarchicalExpertModel and evaluate accuracy on test split."""
    expert_dir = Path(repo_root) / "pysatl_expert"
    csv_path = expert_dir / "expert_ml_dataset_binary.csv"
    json_path = expert_dir / "distribution_families.json"
    model_path = expert_dir / "rf_expert_model.joblib"

    logger.info(f"Loading dataset from: {csv_path}")
    logger.info(f"Loading family mapping from: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        family_map = json.load(f)

    df = pd.read_csv(csv_path)

    features = df.drop(columns=["Target"])
    features = features.replace([np.inf, -np.inf], np.nan).fillna(-1.0)
    features = features.clip(lower=-1.0, upper=10.0)

    X = features
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Dataset split: {len(X_train)} training samples, {len(X_test)} test samples.")

    model = HierarchicalExpertModel(family_map)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    logger.info(f"Overall Test Accuracy: {acc * 100:.2f}%")
    logger.info("Classification Report:\n" + classification_report(y_test, y_pred, digits=4))

    joblib.dump(model, model_path)
    logger.info(f"Successfully saved trained model to '{model_path}'")


if __name__ == "__main__":
    main()
