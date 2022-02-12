from typing import Callable
from numpy import *
import numpy


def derivative(func: Callable[[ndarray], ndarray],
               input: ndarray,
               diff: float = 0.001) -> ndarray:
    '''
    Evaluates the derivative of a function "func" at every element in the "input_" array.
    In other terms, returns dfunc/dinput_
    '''
    horizontal_diff = 2 * diff

    vertical_diff = func(input + numpy.array(diff)) - \
        func(input - numpy.array(diff))

    return vertical_diff/horizontal_diff


def main():
    a = [1, 2, 3]

    b = derivative(numpy.square, a)

    print("derivative(square, {0}={1}".format(a, b))


if "__main__" == __name__:
    main()
