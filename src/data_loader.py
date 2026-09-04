"""Data loading utilities for the Heart Disease Diagnosis project."""
import pandas as pd
from src.config import COLS, TARGET_COL

def load_dataset(file_path: str):

    df = pd.read_csv(file_path)
    df.columns = COLS
    print(df.info())
    return df

def read_csv(file_path: str):
    df = pd.read_csv(file_path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    print(f"{file_path}: X={X.shape}, y={y.shape}")
    return X, y
