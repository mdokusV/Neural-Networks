from enum import Enum
from random import randint
from numpy import ndarray, dot, array
import numpy as np


class VisualEnum(Enum):
    FALSE = 0
    TRUE = 1
    BOTH = 2


VISUAL = VisualEnum.BOTH


def transformation_function(x: ndarray):
    for i in range(len(x)):
        if x[i] > 0:
            x[i] = 1
        elif x[i] < 0:
            x[i] = 0
        else:
            x[i] = randint(0, 1)
    return x


def show_array(list, len_vertical, len_horizontal):
    for i in range(len_vertical):
        to_show = "\t"
        for j in range(len_horizontal):
            match VISUAL:
                case VisualEnum.FALSE:
                    print("x" if list[i * len_horizontal + j] > 0.5 else "o", end="\t")
                case VisualEnum.TRUE:
                    print(round(list[i * len_horizontal + j], 4), end="\t")
                case VisualEnum.BOTH:
                    print(round(list[i * len_horizontal + j], 4), end="\t")
                    to_show += (
                        f"{'x' if list[i * len_horizontal + j] > 0.5 else 'o'}" + "\t"
                    )
        print(to_show)
        print()
    print()


def sum_biased(input: ndarray, weights: ndarray, theta: ndarray) -> ndarray:
    return dot(weights, input) - theta


input = array(
    [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ]
).flatten()

x = array([randint(0, 1) for _ in range(len(input))])
c = np.zeros((len(input), len(input)))
weights = c.copy()
for i in range(len(input)):
    for j in range(len(input)):
        if i != j:
            c[i][j] = (input[i] - 1 / 2) * (input[j] - 1 / 2)
        else:
            c[i][j] = 0
theta = np.zeros(len(input))


def iteration(x: ndarray):

    weights = 2 * c

    for i in range(len(input)):
        theta[i] = sum(c[i])

    value = sum_biased(x, weights, theta)

    return transformation_function(value)


for _ in range(5):
    x = iteration(x)

show_array(x, 5, 5)
