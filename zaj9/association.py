from enum import Enum
from numpy import array, ndarray, outer, where, round
from scipy.__config__ import show


class VisualEnum(Enum):
    FALSE = 0
    TRUE = 1
    BOTH = 2


VISUAL = VisualEnum.BOTH


weight_component: list[ndarray] = []
input: list[ndarray] = []

# region weight component
weight_component.append(
    array(
        [
            [-1, -1, -1, -1, -1],
            [-1, 1, 1, 1, -1],
            [-1, 1, -1, 1, -1],
            [-1, 1, 1, 1, -1],
            [-1, -1, -1, -1, -1],
        ]
    ).flatten()
)
weight_component.append(
    array(
        [
            [-1, -1, -1, -1, -1],
            [-1, 1, 1, -1, -1],
            [-1, -1, 1, -1, -1],
            [-1, -1, 1, -1, -1],
            [-1, -1, -1, -1, -1],
        ]
    ).flatten()
)

# endregion

# region input
input.append(
    array(
        [
            [-1, -1, -1, -1, -1],
            [-1, 1, 1, 1, -1],
            [-1, 1, -1, 1, -1],
            [-1, 1, 1, 1, -1],
            [-1, -1, -1, -1, -1],
        ]
    ).flatten()
)
input.append(
    array(
        [
            [-1, -1, -1, -1, -1],
            [-1, 1, 1, -1, -1],
            [-1, -1, 1, -1, -1],
            [-1, -1, 1, -1, -1],
            [-1, -1, -1, -1, -1],
        ]
    ).flatten()
)
input.append(
    array(
        [
            [-1, 1, 1, 1, -1],
            [-1, 1, -1, 1, -1],
            [-1, 1, -1, 1, -1],
            [-1, 1, 1, 1, -1],
            [-1, -1, -1, -1, -1],
        ]
    ).flatten()
)
input.append(
    array(
        [
            [-1, -1, 1, -1, -1],
            [-1, -1, 1, -1, -1],
            [-1, -1, 1, -1, -1],
            [-1, -1, 1, -1, -1],
            [-1, -1, -1, -1, -1],
        ]
    ).flatten()
)

# endregion


def activation_function(x: ndarray) -> ndarray:
    return where(x >= 0, 1, -1)


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


weights = (
    outer(
        weight_component[0],
        weight_component[0],
    )
    + outer(
        weight_component[1],
        weight_component[1],
    )
) / 25

for i in range(len(input)):
    print(f"Input {i + 1}:")
    show_array(input[i], 5, 5)
    output = activation_function(weights.dot(input[i]))
    print(f"Output {i + 1}:")
    show_array(output, 5, 5)
    print("________________________________________________________")
