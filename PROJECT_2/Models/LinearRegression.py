import pandas as pd
import numpy as np

class LinearRegressionModel:
    def fit(self, X, y):
        """
        Fits the linear regression model using the normal equation.
        """
        X = np.c_[np.ones(X.shape[0]), X] 
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        """
        Predicts using the fitted linear regression model.
        """
        X = np.c_[np.ones(X.shape[0]), X] 
        return X @ self.weights


class Metrics:
    @staticmethod
    def mean_squared_error(y_true, y_pred):

        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def r_squared(y_true, y_pred):
        """
        Calculates R-Squared.
        """
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)
