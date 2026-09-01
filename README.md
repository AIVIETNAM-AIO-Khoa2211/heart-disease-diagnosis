# Heart Disease Diagnosis

Machine learning project for diagnosing heart disease on the Cleveland Heart Disease dataset: benchmarking multiple classifiers and explaining predictions with TreeSHAP.

## Project Structure

```
heart-disease-diagnosis/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── README.md
│   ├── raw/
│   ├── engineered/
│   ├── raw_reduced/
│   └── engineered_reduced/
├── notebooks/
│   ├── README.md
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_benchmark.ipynb
│   ├── 04_treeshap.ipynb
│   └── 05_results.ipynb
├── src/
│   ├── README.md
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── explainability.py
│   └── visualization.py
├── models/
│   └── README.md
├── results/
│   ├── README.md
│   ├── tables/
│   └── figures/
├── app/
│   ├── README.md
│   └── app.py
├── docs/
│   └── README.md
└── tests/
    ├── README.md
    ├── test_preprocessing.py
    ├── test_model_training.py
    └── test_explainability.py
```

## Where to go next

- `data/` — dataset (raw and processed)
- `notebooks/` — step-by-step notebooks (see `notebooks/README.md` for execution order)
- `src/` — reusable Python source code
- `models/` — trained model artifacts (not tracked in git)
- `results/` — experiment outputs: tables and figures
- `app/` — Streamlit demo application
- `docs/` — additional documentation / report
- `tests/` — unit tests

## Getting Started

1. Create a virtual environment and install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Follow the notebooks in `notebooks/` in numerical order.
3. Run the demo app as described in `app/README.md`.

Each folder above has its own `README.md` with further, folder-specific details.
