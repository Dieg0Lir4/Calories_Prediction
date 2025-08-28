import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def LoadData(file_path : str):
    """
    Load data from a CSV file.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    data = pd.read_csv(file_path)
    return data

def CleanData(df : pd.DataFrame):
    """
    Clean the data by eliminating missing values.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame to be cleaned.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df = df.dropna()
    return df

if __name__ == "__main__":
    
    df = LoadData("data/calories.csv")
    df = CleanData(df)
    print(df.describe())
    print(df.info())