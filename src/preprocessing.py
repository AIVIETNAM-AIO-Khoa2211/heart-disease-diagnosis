import pandas as pd
import src.config as config
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def to_num(df, list, target):
    for c in list:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[target] = (df[target] > 0).astype(int)
    return df

def build_preprocessor(
        cate_cols: list = config.CATEGORICAL_COLS,
        num_cols: list = config.NUMERIC_COLS,
) -> ColumnTransformer:
    """Tạo bộ tiền xử lý (chưa fit): điền thiếu + scale cho từng nhóm cột.

    - Cột số: điền thiếu bằng median, chuẩn hóa bằng StandardScaler.
    - Cột phân loại: điền thiếu bằng mode, scale bằng MinMaxScaler.
    """
    num_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    cate_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("scale", MinMaxScaler()),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cate", cate_pipeline, cate_cols),
    ], verbose_feature_names_out=False).set_output(transform="pandas")

    return preprocessor


def scale_data(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        cate_cols: list = config.CATEGORICAL_COLS,
        num_cols: list = config.NUMERIC_COLS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit bộ tiền xử lý CHỈ trên train, rồi transform cả 3 tập.

    Tránh leakage: median/mode/mean/std/min/max chỉ được tính từ train_df,
    sau đó áp dụng y nguyên các con số đó lên val_df và test_df.
    """
    preprocessor = build_preprocessor(cate_cols, num_cols)

    train_scaled = preprocessor.fit_transform(train_df)
    val_scaled = preprocessor.transform(val_df)
    test_scaled = preprocessor.transform(test_df)

    return train_scaled, val_scaled, test_scaled