class Params:
    def __init__(self, thetas, bias):
        self.thetas = thetas
        self.bias = float(bias)
        
class CostHistory:
    def __init__(self, cost_history_train, cost_history_val ,cost_history_test):
        self.cost_history_train = cost_history_train
        self.cost_history_val = cost_history_val
        self.cost_history_test = cost_history_test
        
class DataSplits:
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test):
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

class HyperParams:
    def __init__(self, learning_rate, iterations, min_cost_decrease):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.min_cost_decrease = min_cost_decrease