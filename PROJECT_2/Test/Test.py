import numpy as np 
import pandas as pd 


import sys 
sys.path.insert(1,"E:\IIT CHICAGO\SEMESTER_1\MACHINE LEARNING\PROJECT_2")

from Models.LinearRegression import LinearRegressionModel
from Models.LinearRegression import Metrics
from Models.Kfold import CrossValidation
from Models.Bootstrapping import Bootstrapping
from Data.DataGen import DataGenerator
from Data.DataGen import ProfessorData

generator = ProfessorData(m=[1, -2, 3, 0, 0, 0, 0, 0, 0, 0], N=100, b=5, scale=0.5)
X, y = generator.linear_data_generator()


cv = CrossValidation(k=5)
mse_cv, r2_cv = cv.k_fold_cv(X, y)
print(f"\nAverage k-Fold CV MSE: {mse_cv:.4f}")
print(f"Average k-Fold CV R-Squared: {r2_cv:.4f}")


print("\nBootstrapping Results:")
bs = Bootstrapping(n_iterations=10)  
mse_bs, r2_bs = bs.bootstrap(X, y)
print(f"\nAverage Bootstrap MSE: {mse_bs:.4f}")
print(f"Average Bootstrap R-Squared: {r2_bs:.4f}")