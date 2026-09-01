# Data

To evaluate the impact of preprocessing on tree-based models, the project constructs four dataset variants using the same train/validation/test split:

1. **Original**: The original dataset after imputation and scaling, retaining all 13 features.

2. **FE**: The original dataset augmented with newly engineered features, followed by one-hot encoding of categorical variables and selection of the Top-K features based on mutual information (MI).

3. **Original + DT**: The Original dataset reduced to $K = 10$ features selected according to the feature importance scores provided by the Decision Tree.

4. **FE + DT**: The FE dataset reduced using the same Decision Tree-based feature selection method.
