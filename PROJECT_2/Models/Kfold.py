import numpy as np
import pandas as pd


import sys 
sys.path.insert(1,"E:\IIT CHICAGO\SEMESTER_1\MACHINE LEARNING\PROJECT_2")

from Models.LinearRegression import LinearRegressionModel
from Models.LinearRegression import Metrics

class CrossValidation:
    def __init__(self, k):
        self.k = k

    def k_fold_cv(self, X, y):
        """
        Performs k-fold cross-validation.
        """
        indices = np.arange(len(y))
        np.random.seed(42)
        np.random.shuffle(indices)
        fold_size = len(y) // self.k
        mse_scores = []
        r2_scores = []

        for i in range(self.k):
            val_indices = indices[i * fold_size:(i + 1) * fold_size]
            train_indices = np.setdiff1d(indices, val_indices)

            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]


            model = LinearRegressionModel()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)


            mse_scores.append(Metrics.mean_squared_error(y_val, y_pred))
            r2_scores.append(Metrics.r_squared(y_val, y_pred))

        return np.mean(mse_scores), np.mean(r2_scores)
