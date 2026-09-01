"""Plotting and visualization utilities."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import math

def heatmap(df: pd.DataFrame, cols: list = None, figsize: tuple = (6, 5)) -> None:
    data = df[cols] if cols is not None else df.select_dtypes(include="number")
    corr = data.corr()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Correlation matrix between features")
    plt.tight_layout()
    plt.show()

def inspect_categorical_columns(df, categorical_cols):
    """
    Inspect categorical columns before encoding.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    categorical_cols : list
        List of categorical column names.

    Returns
    -------
    None
    """

    for col in categorical_cols:
        unique_values = df[col].dropna().unique()
        n_unique = len(unique_values)

        print(f"\n{'=' * 50}")
        print(f"Column: {col}")
        print(f"Dtype: {df[col].dtype}")
        print(f"Number of unique values: {n_unique}")
        print("Unique values:")

        for value in unique_values:
            count = (df[col] == value).sum()
            print(f"  - {value!r}: {count} samples")
#Plotting
def plot_categorical_pie_charts(df, cate_cols, ncols=3, figsize_per_plot=(4, 4)):
    """Plot a pie chart of value counts for each categorical column.

    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    cate_cols : list of str
        Categorical column names to plot.
    ncols : int, optional
        Number of subplot columns in the grid (default 3).
    figsize_per_plot : tuple, optional
        (width, height) in inches allocated per subplot (default (4, 4)).
    """
    n = len(cate_cols)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, cate_cols):
        counts = df[col].value_counts()
        ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
        ax.set_title(col)

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_numeric_histograms(df, num_cols, bins=30, ncols=3, figsize_per_plot=(4, 4)):
    """Plot a histogram for each numeric column.

    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    num_cols : list of str
        Numeric column names to plot.
    bins : int, optional
        Number of histogram bins (default 30).
    ncols : int, optional
        Number of subplot columns in the grid (default 3).
    figsize_per_plot : tuple, optional
        (width, height) in inches allocated per subplot (default (4, 4)).
    """
    n = len(num_cols)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, num_cols):
        sns.histplot(df[col].dropna(), bins=bins, kde=True, ax=ax)
        ax.set_title(col)
        ax.set_xlabel(col)

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()