import pandas as pd
import src.config as config
from sklearn.model_selection import train_test_split


def split_train_val_test(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    val_size: float = config.VAL_SIZE,
    target_col: str = config.TARGET_COL,
    random_state: int = config.RANDOM_SEED,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the data into three sets: train, validation, and test.

    The data is split in two steps: the test set is split first using `test_size` as a proportion of the entire DataFrame. 
    The validation set is then split from the remaining train+validation data. 

    For example:
    TEST_SIZE=0.2, VAL_SIZE=0.1
    -> train=70%, val=10%, test=20%

    Args:
    df: DataFrame after feature engineering.
    test_size: Proportion of the entire dataset allocated to the test set. Defaults to config.TEST_SIZE.
    val_size: Proportion of the entire dataset allocated to the validation set. Defaults to config.VAL_SIZE.
    target_col: Name of the target column, used for stratification.
    random_state: Random seed for reproducible splits.
    stratify: If True and the target is a discrete variable, preserve the target class distribution across the three sets.

    Returns:
    A tuple containing (train_df, val_df, test_df).
    """

    if test_size + val_size >= 1.0:
        raise ValueError(f"test_size + val_size phải nhỏ hơn 1.0, hiện tại = {test_size + val_size}")

    def _get_stratify_col(data: pd.DataFrame):
        if stratify and target_col in data.columns and data[target_col].nunique() <= 20:
            return data[target_col]
        return None

    # 1. split train_val and test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=_get_stratify_col(df),
    )

    # 2. Split train and val from the remain data
    relative_val_size = val_size / (1.0 - test_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=_get_stratify_col(train_val_df),
    )

    return train_df, val_df, test_df


def save_train_val_test(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    folder_path,
) -> None:
    """Save the train, validation, and test sets as CSV files in the specified folder.

    Args:
    train_df: DataFrame containing the training set.
    val_df: DataFrame containing the validation set.
    test_df: DataFrame containing the test set.
    folder_path: Directory where the CSV files will be saved.

    Output:
    None. Saves "train.csv", "val.csv", and "test.csv" to `folder_path`.
    """

    folder_path.mkdir(parents=True, exist_ok=True)

    train_path = folder_path / "train.csv"
    val_path = folder_path / "val.csv"
    test_path = folder_path / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    total = len(train_df) + len(val_df) + len(test_df)
    print(f"Đã lưu {len(train_df)} dòng train ({len(train_df)/total:.1%}) vào: {train_path}")
    print(f"Đã lưu {len(val_df)} dòng val ({len(val_df)/total:.1%}) vào: {val_path}")
    print(f"Đã lưu {len(test_df)} dòng test ({len(test_df)/total:.1%}) vào: {test_path}")