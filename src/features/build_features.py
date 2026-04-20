from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


FEATURE_COLS = [
    "Recency",
    "Frequency",
    "Monetary",
    "AvgTicket",
    "CustomerAge",
    "FreqPerDay",
]

OUTLIER_COLS = ["Monetary", "Frequency", "AvgTicket", "FreqPerDay"]


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    recency = (
        df.groupby("CustomerID")["InvoiceDate"]
        .max()
        .reset_index()
    )
    recency["Recency"] = (snapshot_date - recency["InvoiceDate"]).dt.days
    recency = recency[["CustomerID", "Recency"]]

    frequency = (
        df.groupby("CustomerID")["InvoiceNo"]
        .nunique()
        .reset_index()
        .rename(columns={"InvoiceNo": "Frequency"})
    )

    monetary = (
        df.groupby("CustomerID")["TotalPrice"]
        .sum()
        .reset_index()
        .rename(columns={"TotalPrice": "Monetary"})
    )

    customer_age = (
        df.groupby("CustomerID")["InvoiceDate"]
        .min()
        .reset_index()
    )
    customer_age["CustomerAge"] = (snapshot_date - customer_age["InvoiceDate"]).dt.days
    customer_age = customer_age[["CustomerID", "CustomerAge"]]

    features = recency.merge(frequency, on="CustomerID")
    features = features.merge(monetary, on="CustomerID")
    features = features.merge(customer_age, on="CustomerID")

    features["AvgTicket"] = features["Monetary"] / features["Frequency"]
    features["FreqPerDay"] = features["Frequency"] / features["CustomerAge"]

    features = features.replace([np.inf, -np.inf], np.nan).dropna().copy()

    return features


def fit_feature_pipeline(df_features: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df_features = df_features.copy()

    outlier_limits = {}
    filtered = df_features.copy()

    for col in OUTLIER_COLS:
        upper_limit = filtered[col].quantile(0.99)
        outlier_limits[col] = upper_limit
        filtered = filtered[filtered[col] <= upper_limit]

    transformed = filtered.copy()

    for col in FEATURE_COLS:
        transformed[col] = np.log1p(transformed[col])

    scaler = RobustScaler()
    scaled_array = scaler.fit_transform(transformed[FEATURE_COLS])

    processed = pd.DataFrame(
        scaled_array,
        columns=FEATURE_COLS,
        index=transformed.index
    )
    processed.insert(0, "CustomerID", transformed["CustomerID"].values)

    feature_artifact = {
        "feature_columns": FEATURE_COLS,
        "outlier_columns": OUTLIER_COLS,
        "outlier_limits": outlier_limits,
        "scaler": scaler,
    }

    return processed, feature_artifact


def transform_feature_pipeline(df_features: pd.DataFrame, feature_artifact: dict) -> pd.DataFrame:
    df_features = df_features.copy()

    feature_columns = feature_artifact["feature_columns"]
    outlier_columns = feature_artifact["outlier_columns"]
    outlier_limits = feature_artifact["outlier_limits"]
    scaler = feature_artifact["scaler"]

    transformed = df_features.copy()

    # Em inferência, capamos os valores extremos em vez de remover observações
    for col in outlier_columns:
        upper_limit = outlier_limits[col]
        transformed[col] = transformed[col].clip(upper=upper_limit)

    for col in feature_columns:
        transformed[col] = np.log1p(transformed[col])

    transformed = transformed.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_columns).copy()

    if transformed.empty:
        raise ValueError(
            "Nenhuma observação restou após aplicar as transformações de inferência. "
            "Verifique se a base nova contém dados válidos para gerar as features."
        )

    scaled_array = scaler.transform(transformed[feature_columns])

    processed = pd.DataFrame(
        scaled_array,
        columns=feature_columns,
        index=transformed.index
    )
    processed.insert(0, "CustomerID", transformed["CustomerID"].values)

    return processed


def build_features_pipeline(
    df: pd.DataFrame,
    mode: str = "train",
    feature_artifact: dict | None = None
) -> tuple[pd.DataFrame, dict | None]:
    base_features = build_customer_features(df)

    if mode == "train":
        processed_dataset, learned_artifact = fit_feature_pipeline(base_features)
        return processed_dataset, learned_artifact

    if mode == "predict":
        if feature_artifact is None:
            raise ValueError("feature_artifact é obrigatório no modo 'predict'.")

        processed_dataset = transform_feature_pipeline(base_features, feature_artifact)
        return processed_dataset, None

    raise ValueError("mode deve ser 'train' ou 'predict'.")