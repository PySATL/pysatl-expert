"""Hierarchical Clustering Script for Distribution Families in pysatl-expert."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Execute agglomerative clustering to discover distribution families."""
    repo_root = Path(__file__).parents[1]
    expert_dir = repo_root / "pysatl_expert"
    csv_path = expert_dir / "expert_ml_dataset_binary.csv"
    dendrogram_path = repo_root / "distribution_families_dendrogram.png"
    clustermap_path = repo_root / "distribution_clustermap.png"
    json_path = expert_dir / "distribution_families.json"

    logger.info(f"Loading dataset for clustering: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file {csv_path} not found!")

    df = pd.read_csv(csv_path)

    features = df.drop(columns=["Target"])
    features = features.replace([np.inf, -np.inf], np.nan).fillna(-1.0)
    features = features.clip(lower=-1.0, upper=10.0)

    df_clean = pd.concat([features, df["Target"]], axis=1)
    feature_cols = [c for c in df_clean.columns if c != "Target"]
    target_col = "Target"

    profiles = df_clean.groupby(target_col)[feature_cols].mean()
    dist_names = profiles.index.tolist()

    msg = f"Computed profiles for {len(dist_names)} distributions ({len(feature_cols)} features)."
    logger.info(msg)

    Z = linkage(profiles.values, method="ward", metric="euclidean")

    plt.figure(figsize=(10, 6), dpi=300)
    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )

    dendrogram(
        Z,
        labels=dist_names,
        leaf_rotation=0,
        leaf_font_size=12,
        color_threshold=0.7 * max(Z[:, 2]),
    )

    plt.title(
        "Hierarchical Dendrogram of Probability Distribution Families",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Distribution Classes", fontsize=12, labelpad=10)
    plt.ylabel("Ward Distance", fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(dendrogram_path, dpi=300)
    plt.close()
    logger.info(f"Dendrogram plot saved to: '{dendrogram_path}'")

    cg = sns.clustermap(
        profiles,
        row_linkage=Z,
        cmap="coolwarm",
        figsize=(12, 8),
        cbar_kws={"label": "Mean Feature Acceptance Rate"},
        yticklabels=True,
        xticklabels=False,
    )
    cg.fig.suptitle(
        "Clustermap of Distribution Criteria Profiles", fontsize=14, fontweight="bold", y=1.02
    )
    cg.savefig(clustermap_path, dpi=300)
    plt.close()
    logger.info(f"Clustermap saved to: '{clustermap_path}'")

    cluster_ids = fcluster(Z, t=3, criterion="maxclust")

    clusters_dict = {}
    for dist, cid in zip(dist_names, cluster_ids):
        clusters_dict.setdefault(int(cid), []).append(dist)

    formatted_families = {}
    for idx, (cid, members) in enumerate(sorted(clusters_dict.items()), start=1):
        family_name = f"Family_{idx}"
        formatted_families[family_name] = sorted(members)

    logger.info(f"Discovered Distribution Families: {formatted_families}")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(formatted_families, f, indent=2, ensure_ascii=False)

    logger.info(f"Family mapping configuration saved to: '{json_path}'")


if __name__ == "__main__":
    main()
