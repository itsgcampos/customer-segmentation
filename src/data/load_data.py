from pathlib import Path
import pandas as pd


def load_raw_data(file_path: str | Path) -> pd.DataFrame:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    return pd.read_excel(file_path)