import numpy as np


c = 0.1
epsilon = 1e-5
beta = 1.0


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-beta * z))


def sigmoidPrime(z):
    return beta * sigmoid(z) * (1 - sigmoid(z))


def Layer(x: np.ndarray, w: np.ndarray, bias: np.ndarray):
    return sigmoid(np.dot(x, w) + bias)


def gradient(prev_value):
    out = prev_value


def error(real: np.ndarray, predicted: np.ndarray):
    return np.sum(np.abs(real - predicted))


expected = np.array([0, 1, 1, 0])
predicted = np.array([0.5, 0.5, 0.5, 0.5])

print(error(expected, predicted))
