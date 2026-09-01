"""Project-wide configuration: reproducibility seed, data paths, train/val/test
split ratios, feature/target columns, and model hyperparameter templates.
"""
from pathlib import Path

RANDOM_SEED = 42

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
RAW_REDUCED_DIR = DATA_DIR / "raw_reduced"
ENGINEERED_DIR = DATA_DIR / "engineered"
ENGINEERED_REDUCED_DIR = DATA_DIR / "engineered_reduced"

CLEVELAND_CSV_PATH = DATA_DIR / "cleveland.csv"

# Train / validation / test split
TEST_SIZE = 0.1
VAL_SIZE = 0.1
TRAIN_SIZE = 1 - TEST_SIZE - VAL_SIZE  

# Columns
COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

TARGET_COL = "target"

K_FEATURES = 10

# Model hyperparameters
MODEL_PARAMS = {
    "naive_bayes": {
        # sklearn.naive_bayes.GaussianNB
    },
    "knn": {
        # sklearn.neighbors.KNeighborsClassifier
    },
    "decision_tree": {
        # sklearn.tree.DecisionTreeClassifier
        "random_state": RANDOM_SEED,
    },
    "random_forest": {
        # sklearn.ensemble.RandomForestClassifier
        "random_state": RANDOM_SEED,
    },
    "adaboost": {
        # sklearn.ensemble.AdaBoostClassifier
        "random_state": RANDOM_SEED,
    },
    "gradient_boosting": {
        # sklearn.ensemble.GradientBoostingClassifier
        "random_state": RANDOM_SEED,
    },
    "xgboost": {
        # xgboost.XGBClassifier
        "random_state": RANDOM_SEED,
    },
}