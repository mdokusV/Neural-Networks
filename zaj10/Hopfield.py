from random import randint
from numpy import ndarray, dot


def activation_function(x):
    if x > 0:
        return 1
    elif x < 0:
        return 0
    else:
        return randint(0, 1)


def sum_biased(input: ndarray, weights: ndarray, theta: float) -> ndarray:
    return dot(weights, input) - theta
