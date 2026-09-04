import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_classif
from src.config import CATEGORICAL_COLS, RANDOM_SEED
from src.feature_engineer import get_fe_cate_col, is_column_discrete

def select_top_features_by_mi(fe_pipeline: Pipeline, 
                              raw_cate_col: list, 
                              X_train: pd.DataFrame, 
                              y_train: pd.Series, 
                              K = 10, 
                              seed=RANDOM_SEED):

    """
    Select the K most important features based on Mutual Information (MI)
    scores between each column and the target.

    How it works:
    1. Identify which columns in X_train are one-hot encoded categorical
       columns (needed so mutual_info_classif applies the correct formula
       for discrete variables).
    2. Compute the MI score for EVERY column in X_train.
    3. Sort scores in descending order and keep the top K columns.

    Parameters
    ----------
    fe_pipeline : Pipeline
        The FITTED feature engineering pipeline (used to retrieve the
        already-fitted OneHotEncoder, so we know which columns in
        X_train are categorical).
    raw_cate_col : list[str]
        List of ORIGINAL categorical column names (before one-hot
        encoding), e.g. ["sex", "cp", "age_bin"]. NOT the list of all
        columns in X_train.
    X_train : pd.DataFrame
        Training data AFTER feature engineering (the output of
        fe_pipeline). All of X_train's columns are read automatically
        from X_train itself — no need to pass them separately.
    y_train : pd.Series
        Target labels corresponding to X_train.
    K : int, default=10
        Number of top-scoring columns to keep.
    seed : int, default=RANDOM_SEED
        random_state to make the result reproducible.

    Returns
    -------
    topk_cols : list[str]
        Names of the K highest-scoring columns, sorted in descending
        order of MI score.
    mi_series : pd.Series
        MI score for EVERY column in X_train (not just the top K),
        sorted in descending order — useful for plotting or inspecting
        the full ranking.
    """
    
    encoded_cat_cols = get_fe_cate_col(fe_pipeline, raw_cate_cols=raw_cate_col)
    is_discrete = is_column_discrete(X_train.columns, encoded_cat_cols)

    mi = mutual_info_classif(
        X_train.values, y_train.values,
        discrete_features=is_discrete, random_state=seed
    )
    mi_series = pd.Series(mi, index=X_train.columns).sort_values(ascending=False)
    topk_cols = list(mi_series.head(K).index)
    return topk_cols, mi_series