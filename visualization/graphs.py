import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def Histogram(df: pd.DataFrame, column: str):
    """
    Plot a histogram for a specific column in the DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name of the data to plot.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True, bins=30, color="royalblue")
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()