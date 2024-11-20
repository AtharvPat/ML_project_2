import pandas as pd
import numpy as np

import sys 
sys.path.insert(1,"E:\IIT CHICAGO\SEMESTER_1\MACHINE LEARNING\PROJECT_2")

from Models.LinearRegression import LinearRegressionModel
from Models.LinearRegression import Metrics

class Bootstrapping:
    def __init__(self, n_iterations):
        self.n_iterations = n_iterations

    def bootstrap(self, X, y):

        n_samples = len(y)
        mse_scores = []
        r2_scores = []

        for i in range(self.n_iterations):
            bootstrap_indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
            oob_indices = np.setdiff1d(np.arange(n_samples), bootstrap_indices)

            if len(oob_indices) == 0:
                continue

            X_train, X_val = X[bootstrap_indices], X[oob_indices]
            y_train, y_val = y[bootstrap_indices], y[oob_indices]

            model = LinearRegressionModel()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            mse = Metrics.mean_squared_error(y_val, y_pred)
            r2 = Metrics.r_squared(y_val, y_pred)

            mse_scores.append(mse)
            r2_scores.append(r2)

            print(f"Iteration {i + 1}/{self.n_iterations} - MSE: {mse:.4f}, R-Squared: {r2:.4f}")

        return np.mean(mse_scores), np.mean(r2_scores)