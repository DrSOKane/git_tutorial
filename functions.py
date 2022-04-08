import numpy as np

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