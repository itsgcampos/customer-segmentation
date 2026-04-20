from __future__ import annotations
import sys
from pathlib import Path

# adiciona raiz do projeto ao path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from pathlib import Path
import joblib
import pandas as pd

from src.data.load_data import load_raw_data
from src.data.preprocess import process_pipeline
from src.features.build_features import build_features_pipeline
from src.models.predict_cluster import predict_clusters


CLUSTER_LABELS = {
    0: "Clientes Muito Ativos (Low Value)",
    1: "Clientes Valiosos / Engajados",
    2: "Clientes Frequentes de Baixo Valor",
}


def run_test_prediction(
    input_path: str | Path,
    output_path: str | Path,
    artifact_path: str | Path
) -> pd.DataFrame:
    """
    Lê uma base bruta nova, reaplica o pipeline de preprocessing e feature engineering
    usando as regras aprendidas no treino, e atribui clusters.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    artifact_path = Path(artifact_path)

    artifact = joblib.load(artifact_path)
    feature_artifact = artifact["feature_artifact"]

    df_raw = load_raw_data(input_path)
    print(f"Base bruta carregada: {df_raw.shape}")

    df_clean = process_pipeline(df_raw)
    print(f"Base após preprocessamento: {df_clean.shape}")

    df_features, _ = build_features_pipeline(
        df_clean,
        mode="predict",
        feature_artifact=feature_artifact
    )
    print(f"Base de features gerada: {df_features.shape}")

    predictions = predict_clusters(df_features, artifact_path)
    predictions["Segment"] = predictions["Cluster"].map(CLUSTER_LABELS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(f"Resultado salvo em: {output_path}")

    return predictions


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    input_path = project_root / "data" / "raw" / "clients_to_cluster.xlsx"
    output_path = project_root / "predictions" / "clients_clustered.csv"
    artifact_path = project_root / "models" / "kmeans_artifact.pkl"

    result = run_test_prediction(input_path, output_path, artifact_path)
    print(result.head())