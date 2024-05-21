from enum import Enum
from pprint import pprint
from numpy import array, ndarray
import numpy as np


class VisualEnum(Enum):
    FALSE = 0
    TRUE = 1
    BOTH = 2


THRESHOLD = 2.5
VISUAL = VisualEnum.BOTH


def activation_function(x: ndarray) -> ndarray:
    return np.where(x >= THRESHOLD, 1, 0)


input: list[ndarray] = []
weights: list[ndarray] = []
# region input
# input 0
input.append(
    array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )
)

# input 1
input.append(
    array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 1, 1, 0, 0],
        ]
    )
)

# input 2
input.append(
    array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ]
    )
)

# input 3
input.append(
    array(
        [
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )
)

# input 4
input.append(
    array(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ]
    )
)
# endregion

# region weights

# weights 0
weights.append(
    array(
        [
            [1, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
        ]
    )
)
# weights 1
weights.append(
    array(
        [
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 1],
        ]
    )
)
# weights 2
weights.append(
    array(
        [
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ]
    )
)
# weights 3
weights.append(
    array(
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ]
    )
)
# endregion


def convolution(input: ndarray, weights: ndarray) -> ndarray:
    result = np.zeros((len(input), len(input[0])))
    for y_matrix in range(len(input)):
        for x_matrix in range(len(input[0])):
            result[x_matrix][y_matrix] += inside_sum(input, weights, y_matrix, x_matrix)

    result = activation_function(result)

    return result


def show_array(array: ndarray):
    for row in array:
        to_show = "\t"
        for item in row:
            match VISUAL:
                case VisualEnum.FALSE:
                    print("x" if item > 0.5 else "o", end=" ")
                case VisualEnum.TRUE:
                    print(np.round(item, 4), end=" ")
                case VisualEnum.BOTH:
                    print(np.round(item, 4), end=" ")
                    to_show += f"{'x' if item > 0.5 else 'o'}" + " "
        print(to_show)
    print()


def inside_sum(input, weights, y_matrix, x_matrix) -> float:
    value = 0
    for y in range(len(weights)):
        for x in range(len(weights[0])):
            if (0 <= x_matrix + x - 1 < len(input[0])) and (
                0 <= y_matrix + y - 1 < len(input)
            ):
                value += weights[x][y] * input[x_matrix + x - 1][y_matrix + y - 1]
    return value


for i in input:
    print("Input:")
    show_array(i)

    for w in weights:
        print("Weights:")
        show_array(w)
        result = convolution(i, w)
        print("Result:")
        show_array(result)

    print("------------------------------------------------------------")
