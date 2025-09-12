import pandas as pd
import numpy as np

def EraseOutliersByZScore(df: pd.DataFrame, column: str, threshold: float = 3.0):
    """
    Remove outliers from a specific column in the DataFrame using Z-score.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to check for outliers.
    threshold (float): Z-score threshold to identify outliers.

    Returns:
    pd.DataFrame: DataFrame without outliers in the specified column.
    """
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    return df[z_scores < threshold]

def EraseOutliersByIQR(df: pd.DataFrame, column: str, multiplier: float = 1.5):
    """
    Remove outliers from a specific column in the DataFrame using IQR.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to check for outliers.
    multiplier (float): IQR multiplier to identify outliers.

    Returns:
    pd.DataFrame: DataFrame without outliers in the specified column.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[column] >= Q1 - multiplier * IQR) & (df[column] <= Q3 + multiplier * IQR)]

def BinaryMap(df: pd.DataFrame, column: str, firstValue: str, secondValue: str):
    """
    Map a binary column to 0 or 1.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to map.
    firstValue (int): Value to map the first unique value to.
    secondValue (int): Value to map the second unique value to.

    Returns:
    pd.DataFrame: DataFrame with the binary column mapped to specified values.
    """
    df[column] = df[column].map({firstValue: 0, secondValue: 1})
    return df

def EraseOutliersByTemperature(df: pd.DataFrame, max: float = 40.7):
    """
    Remove outliers from the 'Body_Temp' column in the DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    threshold (float): Body_Temp max to identify outliers.

    Returns:
    pd.DataFrame: DataFrame without outliers in the 'Body_Temp' column.
    """
    return df[df['Body_Temp'] <= max]
