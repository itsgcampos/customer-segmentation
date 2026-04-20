from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd


def prepare_new_data(df_features: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    X = df_features.copy()

    if "CustomerID" in X.columns:
        X = X.drop(columns=["CustomerID"])

    X = X.reindex(columns=artifact["feature_columns"])

    if X.isnull().any().any():
        raise ValueError("A base de entrada contém colunas ausentes ou valores nulos após o alinhamento.")

    return X


def predict_clusters(df_features: pd.DataFrame, artifact_path: str | Path) -> pd.DataFrame:
    artifact = joblib.load(artifact_path)
    model = artifact["model"]

    X_prepared = prepare_new_data(df_features, artifact)
    clusters = model.predict(X_prepared)

    result = df_features.copy()
    result["Cluster"] = clusters

    return result


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "processed" / "processed_dataset_v2.csv"
    artifact_path = project_root / "models" / "kmeans_artifact.pkl"

    df_features = pd.read_csv(data_path)
    result = predict_clusters(df_features, artifact_path)

    print(result.head())