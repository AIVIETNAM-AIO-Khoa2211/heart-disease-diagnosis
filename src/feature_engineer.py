import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
 
 
class AddNewFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    We must have the fit and trans because it'll works inside an sklearn Pipeline 
    Creates 3 new "ratio" columns:
         chol_per_age = chol / age
         bps_per_age  = trestbps / age
         hr_ratio     = thalach / age
         (each one is only created if the needed original columns exist)
    """
    def __init__(self, n_age_bins = 5):
        self.n_age_bins = n_age_bins

    def fit(self, X, y=None):
        X = X.copy()
        age_val = pd.to_numeric(X["age"], errors="coerce").dropna()

        # pd.qcut splits the ages into n_age_bins groups that each contain
        # roughly the same NUMBER of people (quantile-based bins).
        # retbins=True also gives us the edge values between groups.
        _, edges = pd.qcut(
            age_val,
            q=self.n_age_bins,
            retbins=True,
            duplicates="drop",  # avoids an error if some edges are identical
        )

        # Extend the first/last edge to -infinity / +infinity so that ANY
        # age value we see later (even outside the training range) still
        # falls inside one of the bins instead of becoming NaN.
        edges[0] = -np.inf
        edges[-1] = np.inf
        self.age_bin_edges_ = edges  # save for use in transform()
 
        return self
    
    def transform(self, X):
        # transform() actually builds the new columns, using the bin edges
        # that were learned in fit(). This can safely be called on
        # train, validation, or test data.
        df = X.copy()
 
        if {"chol", "age"} <= set(df.columns):
            df["chol_per_age"] = df["chol"] / df["age"]
 
        if {"trestbps", "age"} <= set(df.columns):
            df["bps_per_age"] = df["trestbps"] / df["age"]
 
        if {"thalach", "age"} <= set(df.columns):
            df["hr_ratio"] = df["thalach"] / df["age"]
 
        if "age" in df.columns:
            df["age_bin"] = pd.cut(
                df["age"],
                bins=self.age_bin_edges_,
                labels=False,       # use 0, 1, 2, ... instead of range labels
                include_lowest=True,
            ).astype("category")
 
        return df

    def get_feature_names_out(self, input_features=None):
        # This tells sklearn what to call the output columns. Needed so
        # .set_output(transform="pandas") can label the result correctly.
        return list(input_features) + [
            "chol_per_age",
            "bps_per_age",
            "hr_ratio",
            "age_bin",
        ]


def build_pipeline(num_cols, cate_cols, n_age_bin = 5):
    new_num_cols = ["chol_per_age", "bps_per_age", "hr_ratio"]
    new_cate_cols = ["age_bin"]

    all_numeric_cols = num_cols + new_num_cols
    all_categorical_cols = cate_cols + new_cate_cols

    num_pipeline = Pipeline([
        ("impute_missing", SimpleImputer(strategy="median"))
        ("scale", StandardScaler())
    ])
    cate_pipeline = Pipeline([
        ("impute_missing", SimpleImputer(strategy="most_frequent"))
        ("one_hot_encode", OneHotEncoder(handle_unknown="ignore",sparse_output=False))
    ])

    column_transformer = ColumnTransformer([
        ("numeric", num_pipeline, all_numeric_cols),
        ("categorical", cate_pipeline, all_categorical_cols),
    ], verbose_feature_names_out=False).set_output(transform="pandas")
 
    full_pipeline = Pipeline([
        ("add_new_features", AddNewFeaturesTransformer(n_age_bins=n_age_bin)),
        ("preprocess", column_transformer),
    ]).set_output(transform="pandas")
 
    return full_pipeline

def run_feature_engineering(
    X_train,
    X_val,
    X_test,
    y_train,
    numeric_cols,
    categorical_cols,
    n_age_bins=5,
    drop_constant_cols=True,
):
    """
    One-call helper: builds the pipeline, fits it on X_train only,
    then transforms train/val/test.
 
    Parameters
    ----------
    X_train, X_val, X_test : pandas DataFrames
    y_train : target values for the training set (only used for fitting)
    numeric_cols, categorical_cols : list[str]
        Your original column names, same as in build_feature_engineering_pipeline.
    n_age_bins : int
    drop_constant_cols : bool
        If True, remove any output column that only has one unique value
        (it can't help a model, since it never changes).
 
    Returns
    -------
    Xt_train, Xt_val, Xt_test : transformed DataFrames, ready for modeling
    pipeline : the FITTED sklearn Pipeline (keep this if you want to reuse
               it later, e.g. to transform new data or save it with pickle)
    """
    pipeline = build_pipeline(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        n_age_bins=n_age_bins,
    )
 
    # Important: fit_transform is called ONLY on training data.
    # Val/test only ever call .transform(), never .fit(), to avoid leakage.
    Xt_train = pipeline.fit_transform(X_train, y_train)
    Xt_val = pipeline.transform(X_val)
    Xt_test = pipeline.transform(X_test)
 
    if drop_constant_cols:
        useful_cols = Xt_train.columns[Xt_train.nunique(dropna=False) > 1]
        Xt_train = Xt_train[useful_cols]
        Xt_val = Xt_val[useful_cols]
        Xt_test = Xt_test[useful_cols]
 
    print("Number of features after feature engineering:", Xt_train.shape[1])
 
    return Xt_train, Xt_val, Xt_test, pipeline