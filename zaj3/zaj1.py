import random
from typing import Any
import numpy as np
from sympy import Symbol, symbols, diff, exp

EPSILON = 1e-6
ALPHA = 1e-3


def compare(x1: list[float], x2: list[float]):
    for i in range(len(x1)):
        if abs(x1[i] - x2[i]) > EPSILON:
            return False
    return True


def gradient_descent(function, symbols: list[Symbol]):
    number_of_symbols = len(symbols)
    derivative = [diff(function, symbols[i]) for i in range(3)]

    def descent(x1: list[float]) -> list[float]:
        df_sub = 0
        out = []
        for i in x1:
            for j in range(number_of_symbols):
                df_sub = derivative[j].subs(symbols[j], i)
            out.append(df_sub)
        return out

    x_new = [random.uniform(0, 1) for _ in range(3)]
    x_old = [np.inf for _ in range(3)]
    while compare(x_old, x_new):
        x_old = x_new

    return x_new


x1 = symbols("x1")
x2 = symbols("x2")
x3 = symbols("x3")
symbols = [x1, x2, x3]
f1 = 2 * x1**2 + 2 * x2**2 + x3**2 - 2 * x1 * x2 - 2 * x2 * x3 - 2 * x1 + 3
gradient_descent(f1, symbols)

f_prime = diff(f1, x1)
print(f_prime.subs(x1, 0))
