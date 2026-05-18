from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/creditcard.csv")

def load_dataset():
    df = pd.read_csv(DATA_PATH)
    return df