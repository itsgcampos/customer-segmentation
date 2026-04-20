from __future__ import annotations

import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

from src.models.predict_cluster import prepare_new_data


def evaluate_model(df_features: pd.DataFrame, artifact_path: str) -> dict:
    artifact = joblib.load(artifact_path)
    model = artifact["model"]

    X = prepare_new_data(df_features, artifact)
    clusters = model.predict(X)

    sil_score = silhouette_score(X, clusters)
    sil_samples = silhouette_samples(X, clusters)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    evaluation = {
        "n_clusters": artifact["n_clusters"],
        "silhouette_score": sil_score,
        "silhouette_samples": sil_samples,
        "clusters": clusters,
        "cluster_sizes": pd.Series(clusters).value_counts().sort_index(),
        "centroids": pd.DataFrame(model.cluster_centers_, columns=X.columns),
        "cluster_summary": pd.DataFrame(X).assign(Cluster=clusters).groupby("Cluster").mean(),
        "pca_components": X_pca,
        "explained_variance_ratio": pca.explained_variance_ratio_
    }

    return evaluation