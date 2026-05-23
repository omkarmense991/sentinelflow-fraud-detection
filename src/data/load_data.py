# src/data/load_data.py

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data/raw/creditcard.csv"


def load_dataset():
    df = pd.read_csv(DATA_PATH)
    return df
