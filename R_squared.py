import numpy as np
import pandas as pd

def R_squared(fname, function, param1, param2, param3):
    # Outputs R^2 value of an analytic function relative to a dataset
    # fname: name of .csv file where x and y values are stored
    # function: choose between "linear", "quadratic", "exponential" and "Gaussian"
    # param1: gradient if linear, a if quradatic, amplitude if exponential or Gaussian
    # param2: y intercept if linear, b if quadratic, decay constant if exponential, mean if Gaussian
    # param3: c if quadratic, standard deviation if Gaussian

    # Read input data from "fname.csv"
    data = pd.read_csv(fname+".csv", header=None).to_numpy()
    x_values = data[:,0]
    y_values = data[:,1]

    # Obtain fitted values from the chosen function
    if function == "linear":
        y_fitted = linear(x=x_values, m=param1, c=param2)
    elif function == "quadratic":
        y_fitted = quadratic(x=x_values, a=param1, b=param2, c=param3)
    elif function == "exponential":
        y_fitted = exponential(x=x_values, A=param1, k=param2, c=param3)
    elif function == "Gaussian":
        y_fitted = Gaussian(x=x_values, A=param1, mu=param2, sigma=param3)
    elif function == "sinh":
        y_fitted = sinh(x=x_values, A=param1, k=param2, c=param3)
    elif function == "cosh":
        y_fitted = cosh(x=x_values, A=param1, k=param2, c=param3)
    elif function == "tanh":
        y_fitted = tanh(x=x_values, A=param1, k=param2, c=param3)
    else:
        print("The function you selected has not been implemented yet!")
    
    # Calculate R squared value of fit to data
    RSS = 0
    TSS = 0
    mean = np.mean(y_values)
    for i in range(np.size(y_values)):
        RSS = RSS + (y_values[i] - y_fitted[i]) ** 2
        TSS = TSS + (y_values[i] - mean) ** 2
    R_squared = 1 - RSS / TSS
    return R_squared

def linear(x, m, c):
    return m * x + c

def quadratic(x, a, b, c):
    return a * x ** 2 + b * x + c

def exponential(x, A, k, c=0):
    return A * np.exp(k * x) + c

def Gaussian(x, A, mu, sigma, c=0):
    # mu = mean; sigma = standard deviation
    return A * np.exp(-((x - mu) / sigma) ** 2) + c

def sinh(x, A, k, c=0):
    return A * np.sinh(k * x) + c

def cosh(x, A, k, c=0):
    return A * np.cosh(k * x) + c

def tanh(x, A, k, c=0):
    return A * np.tanh(k * x) + c