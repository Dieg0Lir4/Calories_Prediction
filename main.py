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

def PrepareEnviorments(df: pd.DataFrame):
    """
    Prepare the environment by splitting the data into training, validation, and test sets.
    And also reshaping the dataframes to fit the code.

    Parameters:
    df (pd.DataFrame): The input DataFrame to split.

    Returns:
    tuple: Split data into training, validation, and test sets.
    """
    X = df[["Age", "Duration", "Heart_Rate", "Body_Temp"]]
    y = df["Calories"]

    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=42)

    X_train = (X_train - X_train.mean()) / X_train.std()
    X_val = (X_val - X_val.mean()) / X_val.std()
    X_test = (X_test - X_test.mean()) / X_test.std()

    return X_train.to_numpy(), X_val.to_numpy(), X_test.to_numpy(), y_train.to_numpy().reshape(-1,1), y_val.to_numpy().reshape(-1,1), y_test.to_numpy().reshape(-1,1)

def PredictSet(X_val: np.ndarray, y_val: np.ndarray, thetas: np.ndarray, bias: float, set_name: str):
    """
    Make predictions on the validation set and give
    the information about the predictions.

    Parameters:
    X_val (np.ndarray): Validation feature matrix.
    y_val (np.ndarray): Validation target values.
    thetas (np.ndarray): Model parameters.
    bias (float): Model bias.

    Returns:
    np.ndarray: Predicted values for the validation set.
    """
    predictions = X_val.dot(thetas) + bias

    rmse = (sum((predictions - y_val) ** 2) / y_val.shape[0]) ** 0.5
    print(f"{set_name} RMSE:", rmse)

    mae = sum(abs(predictions - y_val)) / y_val.shape[0]
    print(f"{set_name} MAE:", mae)

    r2 = 1 - (sum((predictions - y_val) ** 2) / sum((y_val - y_val.mean()) ** 2))
    print(f"{set_name} R²:", r2)

    mape = sum(abs((predictions - y_val) / y_val)) / y_val.shape[0] * 100
    print(f"{set_name} MAPE:", mape)
    print()

def VisualizeTraining(cost_history_train: list, cost_history_test: list):
    """
    Visualize the cost reduction over iterations.

    Parameters:
    cost_history (list): List of cost values over iterations.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history_train, label='Training Cost', color='royalblue')
    plt.plot(cost_history_test, label='Test Cost', color='orange')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Reduction Over Iterations')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    
    df = LoadData("data/calories.csv")
    df = CleanData(df)
    #VisualizeData(df)

    X = df[["Age", "Duration", "Heart_Rate", "Body_Temp"]]
    y = df["Calories"]

    X = (X - X.mean()) / X.std()
    y = (y - y.mean()) / y.std()

    X_train, X_val, X_test, y_train, y_val, y_test = PrepareEnviorments(df)

    thetas = np.array([1, 1, 1, 1]).reshape(-1,1)
    bias = 1
    learning_rate = 0.001
    iterations = 10000

    thetas, bias, cost_history_train, cost_history_test = gd.GradientDescent(X_train, y_train, thetas, bias, learning_rate, iterations, X_test, y_test)

    PredictSet(X_val, y_val, thetas, bias, "Validation")
    PredictSet(X_test, y_test, thetas, bias, "Test")

    VisualizeTraining(cost_history_train, cost_history_test)
