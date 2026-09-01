"""Data cleaning and feature engineering utilities."""
import pandas as pd
from sklearn.model_selection import train_test_split
def to_num(df, list, target):
    for c in list:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[target] = (df[target] > 0).astype(int)
    return df
