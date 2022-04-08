import numpy as np
import pandas as pd
from functions import linear, quadratic, exponential, Gaussian

def R_squared(fname, function, param1, param2, param3=0):
    data = pd.read_csv(fname, header=None).to_numpy()
    x_values = data[:,0]
    y_values = data[:,1]

    if function == "linear":
        y_fitted = linear(x=x_values, m=param1, c=param2)
    elif function == "quadratic":
        y_fitted = quadratic(x=x_values, a=param1, b=param2, c=param3)
    elif function == "exponential":
        y_fitted = exponential(x=x_values, A=param1, k=param2, c=param3)
    elif function == "Gaussian":
        y_fitted = Gaussian(x=x_values, A=param1, mu=param2, sigma=param3)
    else:
        print("The function you selected has not been implemented yet!")
    
    RSS = 0
    TSS = 0
    mean = np.mean(y_values)
    for i in range(np.size(y_values)):
        RSS = RSS + (y_values[i] - y_fitted[i]) ** 2
        TSS = TSS + (y_values[i] - mean) ** 2
    R_squared = 1 - RSS / TSS
    return R_squared