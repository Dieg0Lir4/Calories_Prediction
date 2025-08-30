import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import algorithms.gradient_descent as gd
import transform.cleaning as cln
import visualization.graphs as graphs
from sklearn.model_selection import train_test_split

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

    X = df[["Age", "Duration", "Heart_Rate", "Body_Temp"]]
    y = df["Calories"]

    X = (X - X.mean()) / X.std()
    y = (y - y.mean()) / y.std()

    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=42)

    X_train = X_train.to_numpy()
    X_val = X_val.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy().reshape(-1,1)
    y_val = y_val.to_numpy().reshape(-1,1)
    y_test = y_test.to_numpy().reshape(-1,1)

    thetas = np.array([1, 1, 1, 1]).reshape(-1,1)
    bias = 1
    learning_rate = 0.0001
    iterations = 10000

    thetas, bias = gd.GradientDescent(X_train, y_train, thetas, bias, learning_rate, iterations)

    y_pred = X_val.dot(thetas) + bias
    
    mse = sum((y_pred - y_val) ** 2) / y_val.shape[0]
    print("Validation MSE:", mse)

    print("Final Thetas:", thetas)
    print("Final Bias:", bias)


