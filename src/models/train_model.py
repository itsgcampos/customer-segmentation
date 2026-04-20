from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def train_kmeans_model(
    df_features: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42,
    n_init: int = 10
) -> dict:

    if "CustomerID" not in df_features.columns:
        raise ValueError("A base de features precisa conter a coluna CustomerID.")

    X = df_features.drop(columns=["CustomerID"]).copy()

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init
    )

    clusters = model.fit_predict(X)
    silhouette = silhouette_score(X, clusters)

    artifact = {
        "model": model,
        "feature_columns": X.columns.tolist(),
        "n_clusters": n_clusters,
        "random_state": random_state,
        "n_init": n_init,
        "silhouette_score": silhouette
    }

    return artifact


def save_artifact(artifact: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "processed" / "processed_dataset_v2.csv"
    model_path = project_root / "models" / "kmeans_artifact.pkl"

    df_features = pd.read_csv(data_path)

    artifact = train_kmeans_model(df_features, n_clusters=3)
    save_artifact(artifact, model_path)

    print(f"Modelo salvo em: {model_path}")
    print(f"Silhouette Score: {artifact['silhouette_score']:.4f}")