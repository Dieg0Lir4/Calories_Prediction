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


def PrepareEnviorments(X: pd.DataFrame, y: pd.DataFrame, z: bool=True):
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

    if z:
        X_train = (X_train - X_train.mean()) / X_train.std()
        X_val = (X_val - X_val.mean()) / X_val.std()
        X_test = (X_test - X_test.mean()) / X_test.std()

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


def ShowParameters(thetas: np.ndarray, bias: float, data_name: str="calories", column_names: list=["Age", "Duration", "Heart_Rate"], method: str="w"):
    """
    Save the model parameters and bias in a txt file.
    
    Parameters:
    thetas (np.ndarray): Model parameters.
    bias (float): Model bias.
    """
    
    
    output_file = f"report/{data_name}_metrics.txt"
    with open(output_file, method) as f:
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

    
    

    column_names = ["Age", "Body_Temp", "Height"]
    X = df[column_names]
    y = df["Calories"]

    data_splits = PrepareEnviorments(X, y)

    #Thetas del tamaño de columns_names
    
    thetas = np.ones((len(column_names), 1))
    bias = 1
    learning_rate = 0.1
    iterations = 10000
    min_cost_decrease = 0.0001
    
    params = classes.Params(thetas, bias)
    cost_history = classes.CostHistory([], [], [])
    hyper_parms = classes.HyperParams(learning_rate, iterations, min_cost_decrease)

    params, cost_history = gd.GradientDescent(data_splits, params, hyper_parms)

    ShowParameters(params.thetas, params.bias, "calories", column_names)
    PredictSet(data_splits.X_train, data_splits.y_train, params.thetas, params.bias, "Training", "calories")
    PredictSet(data_splits.X_val, data_splits.y_val, params.thetas, params.bias, "Validation", "calories")
    PredictSet(data_splits.X_test, data_splits.y_test, params.thetas, params.bias, "Test", "calories")
    
    VisualizeTraining(cost_history.cost_history_train, cost_history.cost_history_val, "Training vs Validation", "Training", "Validation")
    VisualizeTraining(cost_history.cost_history_train, cost_history.cost_history_test, "Training vs Test")
    
    #=====RANDOM FOREST REGRESSOR=====#

    rf = RandomForestRegressor(
        n_estimators=50,      # menos árboles
        max_depth=5,          # profundidad máxima limitada
        min_samples_split=20, # obliga a ramas más grandes
        min_samples_leaf=10,  # hojas con mínimo 10 datos
        random_state=42
    )
    
    X = df.drop(columns=["Calories", "Gender", "Duration"])
    y = df["Calories"]
    
    data_splits_rf = PrepareEnviorments(X, y, False)
    
    rf.fit(data_splits_rf.X_train, data_splits_rf.y_train.ravel())
    y_pred_val = rf.predict(data_splits_rf.X_val)
    y_pred_test = rf.predict(data_splits_rf.X_test)
    
    rmse_val = np.sqrt(np.mean((y_pred_val - data_splits_rf.y_val.ravel()) ** 2))
    mae_val = np.mean(np.abs(y_pred_val - data_splits_rf.y_val.ravel()))
    r2_val = rf.score(data_splits_rf.X_val, data_splits_rf.y_val.ravel())
    
    rmse_test = np.sqrt(np.mean((y_pred_test - data_splits_rf.y_test.ravel()) ** 2))
    mae_test = np.mean(np.abs(y_pred_test - data_splits_rf.y_test.ravel()))
    r2_test = rf.score(data_splits_rf.X_test, data_splits_rf.y_test.ravel())
    
    feature_importances = rf.feature_importances_
    
    output_file = f"report/calories_metrics.txt"
    with open(output_file, "a") as f:
        f.write(f"=== Random Forest Regressor ===\n")
        f.write(f"--- Validation Set ---\n")
        f.write(f"RMSE: {rmse_val:.4f}\n")
        f.write(f"MAE:  {mae_val:.4f}\n")
        f.write(f"R²:   {r2_val:.4f}\n\n")
        f.write(f"--- Test Set ---\n")
        f.write(f"RMSE: {rmse_test:.4f}\n")
        f.write(f"MAE:  {mae_test:.4f}\n")
        f.write(f"R²:   {r2_test:.4f}\n\n")
        f.write("Feature Importances:\n")
        for name, importance in zip(X.columns, feature_importances):
            f.write(f"{name}: {importance:.4f}\n")
        f.write("\n")
    

    #=====IMPROVE CALORIES DATASET=====#

    
    df = cln.EraseOutliersByZScore(df, "Height", 3.0)
    df = cln.EraseOutliersByIQR(df, "Weight", 1.5)
    df = cln.EraseOutliersByTemperature(df, 40.7)
    df = cln.BinaryMap(df, "Gender", "female", "male")
    
    df["Calories_ln"] = np.log(df["Calories"])
    
    column_names = ["Age", "Body_Temp", "Height", "Gender"]
    X = df[column_names]
    y = df["Calories_ln"]

    data_splits = PrepareEnviorments(X, y)
    data_splits.y_val = np.exp(data_splits.y_val)
    data_splits.y_test = np.exp(data_splits.y_test)
    
    thetas = np.ones((len(column_names), 1))
    bias = 1
    learning_rate = 0.1
    iterations = 10000
    min_cost_decrease = 0.0001
    
    params = classes.Params(thetas, bias)
    cost_history = classes.CostHistory([], [], [])
    hyper_parms = classes.HyperParams(learning_rate, iterations, min_cost_decrease)

    params, cost_history = gd.GradientDescent(data_splits, params, hyper_parms)
    
    
    
    
    

    ShowParameters(params.thetas, params.bias, "calories", column_names, "a")
    PredictSet(data_splits.X_train, data_splits.y_train, params.thetas, params.bias, "Training", "calories")
    PredictSet(data_splits.X_val, data_splits.y_val, params.thetas, params.bias, "Validation", "calories")
    PredictSet(data_splits.X_test, data_splits.y_test, params.thetas, params.bias, "Test", "calories")
    
    VisualizeTraining(cost_history.cost_history_train, cost_history.cost_history_val, "Training vs Validation Improved", "Training", "Validation")
    VisualizeTraining(cost_history.cost_history_train, cost_history.cost_history_test, "Training vs Test Improved")
    
    
    #=====RANDOM FOREST REGRESSOR IMPROVED=====#
    
    top_features = ["Heart_Rate", "Age", "Gender","Body_Temp"]
    X = df[top_features]
    y = df["Calories"]
    
    rf = RandomForestRegressor(
        n_estimators=800,    # muchos árboles para estabilidad
        max_depth=None,      # sin límite, que cada árbol crezca
        min_samples_split=2, # splits muy pequeños permitidos
        min_samples_leaf=1,  # hojas con 1 dato (máxima fineza)
        random_state=42,
        n_jobs=-1
    )
    
    data_splits_rf = PrepareEnviorments(X, y, False)
    rf.fit(data_splits_rf.X_train, data_splits_rf.y_train.ravel())
    y_pred_val = rf.predict(data_splits_rf.X_val)
    y_pred_test = rf.predict(data_splits_rf.X_test)
    rmse_val = np.sqrt(np.mean((y_pred_val - data_splits_rf.y_val.ravel()) ** 2))
    mae_val = np.mean(np.abs(y_pred_val - data_splits_rf.y_val.ravel()))
    r2_val = rf.score(data_splits_rf.X_val, data_splits_rf.y_val.ravel())
    rmse_test = np.sqrt(np.mean((y_pred_test - data_splits_rf.y_test.ravel()) ** 2))
    mae_test = np.mean(np.abs(y_pred_test - data_splits_rf.y_test.ravel()))
    r2_test = rf.score(data_splits_rf.X_test, data_splits_rf.y_test.ravel())
    feature_importances = rf.feature_importances_
    
    output_file = f"report/calories_metrics.txt"
    with open(output_file, "a") as f:
        f.write(f"=== Random Forest Regressor Improved ===\n")
        f.write(f"--- Validation Set ---\n")
        f.write(f"RMSE: {rmse_val:.4f}\n")
        f.write(f"MAE:  {mae_val:.4f}\n")
        f.write(f"R²:   {r2_val:.4f}\n\n")
        f.write(f"--- Test Set ---\n")
        f.write(f"RMSE: {rmse_test:.4f}\n")
        f.write(f"MAE:  {mae_test:.4f}\n")
        f.write(f"R²:   {r2_test:.4f}\n\n")
        f.write("Feature Importances:\n")
        for name, importance in zip(X.columns, feature_importances):
            f.write(f"{name}: {importance:.4f}\n")
        f.write("\n")
        f.close()
