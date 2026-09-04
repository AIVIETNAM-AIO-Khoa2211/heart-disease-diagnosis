import pandas as pd
import src.config as config
from sklearn.tree import DecisionTreeClassifier

def  select_by_dt(X_train: pd.DataFrame,
                  y_train: pd.Series, 
                  k=10, prefix="dt"):
    dt = DecisionTreeClassifier(random_state=config.RANDOM_SEED)
    dt.fit(X_train,y_train)

    importance = pd.Series(
        dt.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)

    select_cols = importance.head(k).index.to_list()
    print(f"Top {k} đặc trưng ({prefix}): {select_cols}")

    return importance, select_cols
        