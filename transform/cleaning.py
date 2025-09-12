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


def GetDummiesUsingSemicolon(df: pd.DataFrame, column: str):
    """
    Convert a column with semicolon-separated values into multiple binary columns.
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name containing semicolon-separated values.
    Returns:
    pd.DataFrame: DataFrame with new binary columns for each unique value.
    """

    platform_dummies = df["platforms"].str.get_dummies(sep=";")
    df = pd.concat([df, platform_dummies], axis=1)
    
    return df

def GetTopDummiesCorrelation(df: pd.DataFrame, columns: list, target: str, top_n: int = 5, colineal_threshold: float = 0.1):
    """
    Select top N columns based on their correlation with the target variable,
    while avoiding colinearity among the selected columns.
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of column names to consider for correlation.
    target (str): Target variable name.
    top_n (int): Number of top correlated columns to select.
    colineal_threshold (float): Threshold to determine colinearity between columns.
    Returns:
    pd.DataFrame: DataFrame with selected top N correlated columns added.
    list: List of selected column names.
    """
    
    
    df_dummies = pd.DataFrame()
    for column in columns:
        dummies = df[column].str.get_dummies(sep=";")
        dummies = dummies.add_prefix(f"{column}_")
        df_dummies = pd.concat([df_dummies, dummies], axis=1)
    
    df_dummies = pd.concat([df[target], df_dummies], axis=1)
        
        

    correlations = df_dummies.corr()[target].abs().sort_values(ascending=False)
    correlations = correlations[correlations.index != target]
    

    correlations = correlations.sort_values(ascending=False)
    

    selected_columns = []
    for col in correlations.index:
        es_colineal = False
        for sel_col in selected_columns:
            if abs(df_dummies[col].corr(df_dummies[sel_col])) > colineal_threshold:
                es_colineal = True
                break
        if not es_colineal:
            selected_columns.append(col)
        if len(selected_columns) >= top_n:
            break
    
    df = pd.concat([df, df_dummies[selected_columns]], axis=1)
    return df, selected_columns
    