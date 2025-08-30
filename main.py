import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import algorithms.gradient_descent as gd
import transform.cleaning as cln
import visualization.graphs as graphs

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
    Clean the data by using specific techniques depending on the column.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame to be cleaned.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df = cln.EraseOutliersByZScore(df, "Height", 3.0)
    df = cln.EraseOutliersByIQR(df, "Weight", 1.5)
    df = cln.BinaryMap(df, "Gender", "female", "male")

    return df

def VisualizeData(df: pd.DataFrame):
    """
    Visualize the cleaned data using various plots.

    Parameters:
    df (pd.DataFrame): The input DataFrame to visualize.
    """
    graphs.Histogram(df, "Age")
    graphs.Histogram(df, "Height")
    graphs.Histogram(df, "Weight")


if __name__ == "__main__":
    
    df = LoadData("data/calories.csv")
    df = CleanData(df)
    #VisualizeData(df)
    df = df[["Age", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
    print(df.shape)


    

    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6],
        "C": [7, 8, 9]
    })
    
    X = np.array([[10, 20, 30], [10, 17, 25], [5, 50, 40], [1, 2, 3]]).reshape(-1,3)
    y = np.array([7, 25, 9, 18]).reshape(-1,1)
    b = 100
    thetas = np.array([6, 10, 1]).reshape(3,-1)

    bias = 100
    learning_rate = 0.001
    iterations = 10
    gd.GradientDescent(X, y, thetas, bias, learning_rate, iterations)

    