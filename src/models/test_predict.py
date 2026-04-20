from pathlib import Path
import pandas as pd

from src.data.load_data import load_raw_data
from src.data.preprocess import process_pipeline
from src.features.build_features import build_features_pipeline
from src.models.predict_cluster import predict_clusters


def run_test_prediction(input_path: str | Path, output_path: str | Path, artifact_path: str | Path) -> pd.DataFrame:
    input_path = Path(input_path)
    output_path = Path(output_path)
    artifact_path = Path(artifact_path)

    df_raw = load_raw_data(input_path)
    print(f"Base bruta carregada: {df_raw.shape}")

    df_clean = process_pipeline(df_raw)
    print(f"Base após preprocessamento: {df_clean.shape}")

    df_features, _ = build_features_pipeline(df_clean)
    print(f"Base de features gerada: {df_features.shape}")

    predictions = predict_clusters(df_features, artifact_path)
    print(f"Base com clusters: {predictions.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(f"Resultado salvo em: {output_path}")

    return predictions


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    input_path = project_root / "data" / "raw" / "clients_to_cluster.xlsx"
    output_path = project_root / "predictions" / "clusters" / "clients_clustered.csv"
    artifact_path = project_root / "models" / "kmeans_artifact.pkl"

    result = run_test_prediction(input_path, output_path, artifact_path)
    print(result.head())