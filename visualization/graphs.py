import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

#Function to plot histogram of predictions errors meaning to see if prediction high or low

def HistogramOfPredictions(y_true: pd.Series, y_pred: np.ndarray, error: pd.Series = None, name: str = "Prediction Errors", path: str = None):
    """
    Plot a histogram of prediction errors.

    Parameters:
    y_true (pd.Series): Actual target values.
    y_pred (pd.Series): Predicted target values.
    """
    if error is None:
        error = y_pred - y_true
    plt.figure(figsize=(10, 6))
    sns.histplot(error, bins=30, kde=True)
    plt.title(name)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    if path is not None:
        plt.savefig(path)
    plt.close()