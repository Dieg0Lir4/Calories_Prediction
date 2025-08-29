import numpy as np

def Hypothesis(X: np.ndarray, thetas: np.ndarray, bias: float):
    """
    Calculate the hypothesis for linear regression.
    
    Parameters:
    X (np.ndarray): Feature matrix.
    thetas (np.ndarray): Model parameters.
    
    Returns:
    np.ndarray: Predicted values.
    """ 
    result = np.zeros((X.shape[0], 1))
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            result[i,0] += X[i,j] * thetas[j,0]
    
    return result + bias

def MinSquareError(X: np.ndarray, y: np.ndarray, thetas: np.ndarray, bias: float):
    """
    Calculate the Mean Squared Error (MSE) cost function.
    
    Parameters:
    X (np.ndarray): Feature matrix.
    y (np.ndarray): Actual target values.
    thetas (np.ndarray): Model parameters.
    
    Returns:
    float: Computed MSE cost.
    """
    m = y.shape[0]
    predictions = Hypothesis(X, thetas, bias)
    cost = (1/(2*m)) * np.sum((predictions - y) ** 2)
    return cost

def UpdateParameters(X: np.ndarray, y: np.ndarray, thetas: np.ndarray, bias: float, learning_rate: float):
    """
    Update model parameters using gradient descent.
    
    Parameters:
    X (np.ndarray): Feature matrix.
    y (np.ndarray): Actual target values.
    thetas (np.ndarray): Model parameters.
    learning_rate (float): Learning rate for gradient descent.
    
    Returns:
    tuple: Updated thetas and bias.
    """
    m = y.shape[0]
    predictions = Hypothesis(X, thetas, bias)
    error = predictions - y
    sum_errors = np.zeros((X.shape[1], 1))
     
    for i in range(X.shape[1]):
        for j in range(X.shape[0]):
            sum_errors[i,0] += X[j,i] * error[j,0]
            
    thetas = thetas - (learning_rate/m) * sum_errors
    bias = bias - (learning_rate/m) * sum(error)
    return thetas, bias


def GradientDescent(X: np.ndarray, y: np.ndarray, thetas: np.ndarray, bias: float, learning_rate: float, iterations: int):
    """
    Perform gradient descent to optimize model parameters.
    
    Parameters:
    X (np.ndarray): Feature matrix.
    y (np.ndarray): Actual target values.
    thetas (np.ndarray): Initial model parameters.
    learning_rate (float): Learning rate for gradient descent.
    iterations (int): Number of iterations for gradient descent.
    
    Returns:
    tuple: Optimized model parameters and bias.
    """
    for _ in range(iterations):
        thetas, bias = UpdateParameters(X, y, thetas, bias, learning_rate)
        cost = MinSquareError(X, y, thetas, bias)
        if cost < 0.001:
            break
    
    print(f"Parameters after training: Thetas: {thetas}, Bias: {bias}")
    return thetas, bias

