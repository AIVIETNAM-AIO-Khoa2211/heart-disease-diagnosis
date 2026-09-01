# Data
## Folder's structure
To evaluate the impact of preprocessing on tree-based models, the project constructs four dataset variants using the same train/validation/test split:

1. **Original**: The original dataset after imputation and scaling, retaining all 13 features.

2. **FE**: The original dataset augmented with newly engineered features, followed by one-hot encoding of categorical variables and selection of the Top-K features based on mutual information (MI).

3. **Original + DT**: The Original dataset reduced to $K = 10$ features selected according to the feature importance scores provided by the Decision Tree.

4. **FE + DT**: The FE dataset reduced using the same Decision Tree-based feature selection method.

## Raw feature description

This database contains **13 attributes** and **1 target variable**. It has **8 nominal features** and **5 numerical features**. The detailed description of all features is provided below.

### 1. Age
- **Feature:** `age`
- **Description:** Patient's age in years.
- **Type:** Numerical

### 2. Sex
- **Feature:** `sex`
- **Description:** Patient's gender.
  - `1`: Male
  - `0`: Female
- **Type:** Nominal

### 3. Chest Pain Type
- **Feature:** `cp`
- **Description:** Type of chest pain experienced by the patient.
  - `0`: Typical angina
  - `1`: Atypical angina
  - `2`: Non-anginal pain
  - `3`: Asymptomatic
- **Type:** Nominal

### 4. Resting Blood Pressure
- **Feature:** `trestbps`
- **Description:** Patient's resting blood pressure, measured in mm/Hg.
- **Type:** Numerical

### 5. Serum Cholesterol
- **Feature:** `chol`
- **Description:** Serum cholesterol level, measured in mg/dl.
- **Type:** Numerical

### 6. Fasting Blood Sugar
- **Feature:** `fbs`
- **Description:** Fasting blood sugar level. A value greater than 120 mg/dl is represented as:
  - `1`: True
  - `0`: False
- **Type:** Nominal

### 7. Resting Electrocardiographic Results
- **Feature:** `restecg`
- **Description:** Results of the electrocardiogram at rest.
  - `0`: Normal
  - `1`: ST-T wave abnormality (T wave inversions and/or ST elevation or depression > 0.05 mV)
  - `2`: Probable or definite left ventricular hypertrophy according to Estes' criteria
- **Type:** Nominal

### 8. Maximum Heart Rate Achieved
- **Feature:** `thalach`
- **Description:** Maximum heart rate achieved by the patient.
- **Type:** Numerical

### 9. Exercise-Induced Angina
- **Feature:** `exang`
- **Description:** Whether angina was induced by exercise.
  - `0`: No
  - `1`: Yes
- **Type:** Nominal

### 10. ST Depression
- **Feature:** `oldpeak`
- **Description:** Exercise-induced ST depression relative to the state of rest.
- **Type:** Numerical

### 11. Slope
- **Feature:** `slope`
- **Description:** The slope of the ST segment during peak exercise.
  - `0`: Upsloping
  - `1`: Flat
  - `2`: Downsloping
- **Type:** Nominal

### 12. Number of Major Vessels
- **Feature:** `ca`
- **Description:** Number of major vessels, ranging from 0 to 3.
- **Type:** Nominal

### 13. Thalassemia
- **Feature:** `thal`
- **Description:** Blood disorder called thalassemia.
  - `0`: Null
  - `1`: Normal blood flow
  - `2`: Fixed defect (no blood flow in some part of the heart)
  - `3`: Reversible defect (blood flow is observed but is not normal)
- **Type:** Nominal

## Target Variable

### 14. Target
- **Feature:** `target`
- **Description:** The target variable to be predicted.
  - `0`: Patient is normal / does not have heart disease
  - `1`: Patient is suffering from heart disease
- **Type:** Binary
