from math import e
import numpy as np

BETA_SPACE = np.linspace(1, 3, 5)


def f(x):
    return 1 / (1 + np.exp(-BETA * x))


def first_layer(x: np.ndarray, w: np.ndarray) -> list:
    out_vector = [0 for _ in range(2)] + [1]
    for i in range(2):
        out_vector[i] = f(sum([x[j] * w[i][j] for j in range(3)]))

    return out_vector


def last_layer(x: list, s: np.ndarray) -> float:
    return f(sum([x[j] * s[j] for j in range(3)]))


w = np.array([[2, 2, -3], [2, 2, -1]])
s = np.array([-2, 2, -1])
example = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

for BETA in BETA_SPACE:
    print(f"\n{BETA}")
    for x in example:
        first_layer_output = first_layer(x, w)
        last_layer_output = last_layer(first_layer_output, s)
        print(x, last_layer_output)
