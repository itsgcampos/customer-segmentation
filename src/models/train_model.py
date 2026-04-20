from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.data.load_data import load_raw_data
from src.data.preprocess import process_pipeline
from src.features.build_features import build_features_pipeline


def train_kmeans_model(
    df_features: pd.DataFrame,
    feature_artifact: dict,
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
        "silhouette_score": silhouette,
        "feature_artifact": feature_artifact,
    }

    return artifact


def save_artifact(artifact: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    raw_path = project_root / "data" / "raw" / "customer_segmentation.xlsx"
    model_path = project_root / "models" / "kmeans_artifact.pkl"
    processed_path = project_root / "data" / "processed" / "processed_dataset_v2.csv"

    df_raw = load_raw_data(raw_path)
    df_clean = process_pipeline(df_raw)
    df_features, feature_artifact = build_features_pipeline(df_clean, mode="train")
    df_features.to_csv(processed_path, index=False)

    artifact = train_kmeans_model(
        df_features=df_features,
        feature_artifact=feature_artifact,
        n_clusters=3
    )
    save_artifact(artifact, model_path)

    print(f"Dataset processado salvo em: {processed_path}")
    print(f"Modelo salvo em: {model_path}")
    print(f"Silhouette Score: {artifact['silhouette_score']:.4f}")