import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import algorithms.gradient_descent as gd
import transform.cleaning as cln
import visualization.graphs as graphs
import objetos.objetos as classes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


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


def PrepareEnviorments(X: pd.DataFrame, y: pd.DataFrame):
    """
    Prepare the environment by splitting the data into training, validation, and test sets.
    And also reshaping the dataframes to fit the code.

    Parameters:
    df (pd.DataFrame): The input DataFrame to split.

    Returns:
    DataSplits: Split data into training, validation, and test sets.
    """

    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=42)

    #X_train = (X_train - X_train.mean()) / X_train.std()
    #X_val = (X_val - X_val.mean()) / X_val.std()
    #X_test = (X_test - X_test.mean()) / X_test.std()

    data_splits = classes.DataSplits(X_train.to_numpy(),
                                     X_val.to_numpy(),
                                     X_test.to_numpy(),
                                     y_train.to_numpy().reshape(-1,1),
                                     y_val.to_numpy().reshape(-1,1),
                                     y_test.to_numpy().reshape(-1,1))
    
    return data_splits

def PredictSet(X_val: np.ndarray, y_val: np.ndarray, thetas: np.ndarray, bias: float, set_name: str, data_name: str="calories"):
    """
    Make predictions on the set and give
    the information about the predictions in a txt format.

    Parameters:
    X_val (np.ndarray): Validation feature matrix.
    y_val (np.ndarray): Validation target values.
    thetas (np.ndarray): Model parameters.
    bias (float): Model bias.
    set_name (str): Name of the dataset (e.g., "Validation", "Test").
    """
    predictions = X_val.dot(thetas) + bias

    rmse = (sum((predictions - y_val) ** 2) / y_val.shape[0]) ** 0.5
    mae = sum(abs(predictions - y_val)) / y_val.shape[0]
    r2 = 1 - (sum((predictions - y_val) ** 2) / sum((y_val - y_val.mean()) ** 2))
    
    rmse, mae, r2 = rmse.item(), mae.item(), r2.item()

    output_file = f"report/{data_name}_metrics.txt"
    with open(output_file, "a") as f:
        f.write(f"=== {set_name} ===\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAE:  {mae:.4f}\n")
        f.write(f"R²:   {r2:.4f}\n\n")


def ShowParameters(thetas: np.ndarray, bias: float, data_name: str="calories", column_names: list=["Age", "Duration", "Heart_Rate", "Body_Temp"]):
    """
    Save the model parameters and bias in a txt file.
    
    Parameters:
    thetas (np.ndarray): Model parameters.
    bias (float): Model bias.
    """
    
    
    output_file = f"report/{data_name}_metrics.txt"
    with open(output_file, "w") as f:
        f.write("=== Model Parameters ===\n")
        for i, theta in enumerate(thetas):
            f.write(f"Theta {i} ({column_names[i]}): {theta[0]:.6f}\n")
        f.write(f"Bias: {bias:.6f}\n\n")
        

def VisualizeTraining(cost_history_a: list, cost_history_b: list, plot_name: str, set_1:str="Training", set_2:str="Test"):
    """
    Visualize the cost reduction over iterations.

    Parameters:
    cost_history_a (list): List of cost values over iterations.
    cost_history_b (list): List of cost values over iterations.
    plot_name (str): Name for the plot file.
    """
    sns.set_theme(style="whitegrid", context="notebook")

    plt.figure(figsize=(8, 5))
    plt.plot(cost_history_a, label=f"{set_1} Cost", linewidth=2)
    plt.plot(cost_history_b, label=f"{set_2} Cost", linewidth=2, color="orange")

    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title(f"Cost Reduction Over Iterations - {plot_name}")
    plt.legend()
    plt.tight_layout()

    outdir = "report"
    path = os.path.join(outdir, f"{plot_name}_cost_calories.png")
    plt.savefig(path, dpi=200)
    plt.close()


if __name__ == "__main__":
    
    #=====CALORIES DATASET=====#
    
    df = LoadData("data/calories.csv")

    
    df = cln.EraseOutliersByZScore(df, "Height", 3.0)
    df = cln.EraseOutliersByIQR(df, "Weight", 1.5)
    df = cln.EraseOutliersByTemperature(df, 40.7)
    df = cln.BinaryMap(df, "Gender", "female", "male")

    X = df[["Age", "Duration", "Heart_Rate", "Body_Temp"]]
    y = df["Calories"]

    data_splits = PrepareEnviorments(X, y)

    thetas = np.array([1, 1, 1, 1]).reshape(-1,1)
    bias = 1
    learning_rate = 0.1
    iterations = 10000
    min_cost_decrease = 0.0001
    
    params = classes.Params(thetas, bias)
    cost_history = classes.CostHistory([], [], [])
    hyper_parms = classes.HyperParams(learning_rate, iterations, min_cost_decrease)

    params, cost_history = gd.GradientDescent(data_splits, params, hyper_parms)

    ShowParameters(params.thetas, params.bias)
    PredictSet(data_splits.X_train, data_splits.y_train, params.thetas, params.bias, "Training", "calories")
    PredictSet(data_splits.X_val, data_splits.y_val, params.thetas, params.bias, "Validation", "calories")
    PredictSet(data_splits.X_test, data_splits.y_test, params.thetas, params.bias, "Test", "calories")
    
    VisualizeTraining(cost_history.cost_history_train, cost_history.cost_history_val, "Training vs Validation", "Training", "Validation")
    VisualizeTraining(cost_history.cost_history_train, cost_history.cost_history_test, "Training vs Test")
    
    #=====RANDOM FOREST REGRESSOR=====#

    # Using the same data splits from above
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=100,             # limita profundidad
        min_samples_split=15,      # evita ramas muy chicas
        min_samples_leaf=10,       # evita hojas con 1 dato
        random_state=42,
        n_jobs=-2
    )
    rf.fit(data_splits.X_train, data_splits.y_train.ravel())
    y_pred = rf.predict(data_splits.X_test)

    # Métricas
    rmse = np.sqrt(np.mean((y_pred - data_splits.y_test.ravel()) ** 2))
    mae = np.mean(np.abs(y_pred - data_splits.y_test.ravel()))
    r2 = 1 - (np.sum((y_pred - data_splits.y_test.ravel()) ** 2) /
              np.sum((data_splits.y_test.ravel() - data_splits.y_test.ravel().mean()) ** 2))

    # Guardar en archivo
    output_file = f"report/calories_metrics.txt"
    with open(output_file, "a") as f:
        f.write(f"=== Random Forest Regressor ===\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAE:  {mae:.4f}\n")
        f.write(f"R²:   {r2:.4f}\n")
        f.write("\nParámetros del modelo:\n")
        for param, value in rf.get_params().items():
            f.write(f" - {param}: {value}\n")
        f.write("\n")
    
    #=====RAW STEAM DATASET=====#
    """_summary_
    df = LoadData("data/steam.csv")
    
    df = df.dropna()
    owners_split = df["owners"].str.split("-", expand=True).astype(int)
    df["owners"] = ((owners_split[0] + owners_split[1]) // 2)
    
    df = cln.GetDummiesUsingSemicolon(df, "platforms")
    
    df["positive_ratio"] = (df["positive_ratings"] / (df["positive_ratings"] + df["negative_ratings"]))*100
    
    dummies_list = ["categories", "genres", "steamspy_tags"]
    columns = []
    df, columns = cln.GetTopDummiesCorrelation(df, dummies_list, "positive_ratio", top_n=5, colineal_threshold=0.5)
    
    total_columns = columns + ["price", "owners", "average_playtime"]
    
    X = df[total_columns]
    y = df["positive_ratio"]
    
    data_splits = PrepareEnviorments(X, y)
    
    print(data_splits.X_train.shape)
    thetas = np.array([1] * data_splits.X_train.shape[1]).reshape(-1,1)
    print("thetas shape:", thetas.shape)
    bias = 1
    learning_rate = 0.01
    iterations = 10
    min_cost_decrease = 0.0001
    
    params = classes.Params(thetas, bias)
    cost_history = classes.CostHistory([], [], [])
    hyper_parms = classes.HyperParams(learning_rate, iterations, min_cost_decrease)
    params, cost_history = gd.GradientDescent(data_splits, params, hyper_parms)
    
    ShowParameters(params.thetas, params.bias, "raw_steam", total_columns)
    PredictSet(data_splits.X_train, data_splits.y_train, params.thetas, params.bias, "Training", "raw_steam")
    PredictSet(data_splits.X_val, data_splits.y_val, params.thetas, params.bias, "Validation", "raw_steam")
    PredictSet(data_splits.X_test, data_splits.y_test, params.thetas, params.bias, "Test", "raw_steam")
    VisualizeTraining(cost_history.cost_history_train, cost_history.cost_history_val, "Training vs Validation Raw Steam", "Training", "Validation")
    VisualizeTraining(cost_history.cost_history_train, cost_history.cost_history_test, "Training vs Test Raw Steam", "Training", "Test")
    """
    


    

    
