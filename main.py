from pathlib import Path
import pandas as pd

from src.data.load_data import load_raw_data
from src.data.preprocess import process_pipeline
from src.features.build_features import build_features_pipeline
from src.models.train_model import train_kmeans_model, save_artifact


PROJECT_ROOT = Path(__file__).resolve().parent


def run_data_pipeline() -> pd.DataFrame:
    raw_path = PROJECT_ROOT / "data" / "raw" / "cutomer_segmentation.xlsx"
    processed_path = PROJECT_ROOT / "data" / "processed" / "processed_dataset_v2.csv"

    df_raw = load_raw_data(raw_path)
    df_clean = process_pipeline(df_raw)
    df_features, feature_artifact = build_features_pipeline(df_clean, mode="train")

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(processed_path, index=False)

    print(f"Dataset processado salvo em: {processed_path}")
    return df_features


def run_training_pipeline(df_features: pd.DataFrame | None = None) -> dict:
    if df_features is None:
        processed_path = PROJECT_ROOT / "data" / "processed" / "processed_dataset_v2.csv"
        df_features = pd.read_csv(processed_path)

    artifact = train_kmeans_model(df_features, n_clusters=3)

    model_path = PROJECT_ROOT / "models" / "kmeans_artifact.pkl"
    save_artifact(artifact, model_path)

    print(f"Modelo salvo em: {model_path}")
    print(f"Silhouette Score: {artifact['silhouette_score']:.4f}")

    return artifact


if __name__ == "__main__":
    df_features = run_data_pipeline()
    run_training_pipeline(df_features)