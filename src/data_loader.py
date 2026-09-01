"""Data loading utilities for the Heart Disease Diagnosis project."""
import pandas as pd
from src.config import COLS

def load_dataset(file_path: str):

    df = pd.read_csv(file_path)
    df.columns = COLS
    print(df.info())
    return df