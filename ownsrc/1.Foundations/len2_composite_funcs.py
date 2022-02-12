
from .derivatives import *
from typing import Callable
import numpy as np
from numpy import *
from typing import *

import matplotlib
import matplotlib.pyplot as plt

Function = Callable[[ndarray], ndarray]
ChainFunc = List[Function]


def composite_len2_func(chain_func: ChainFunc,
                        input_: ndarray) -> ndarray:
    '''Implementation of a length two composite function x -> f2(f1(x))'''
    f1 = chain_func[0]
    f2 = chain_func[1]

    return f2(f1(input_))


def composite_len2_func_deriv(chain_func: ChainFunc,
                              input: ndarray,
                              diff: float = 0.001) -> ndarray:
    '''
    Implementation of the derivative of a length two composite function :
    (f2(f1))'(x) = f2'(f1(x)) * f1'(x)
    '''
    f1 = chain_func[0]
    f2 = chain_func[1]

    return derivative(f2, f1(input)) * derivative(f1, input)


def plot_composite_func(ax,
                        func,
                        chain_subfuncs: ChainFunc,
                        input: ndarray) -> None:
    '''Draw plot for a composite function'''
    ax.plot(input, func(chain_subfuncs, input))


def plot_funcs_and_derivs():
    PLOT_RANGE = np.arange(-3, 3, 0.01)
    chain1 = [np.square, lambda x: 1/(1+np.exp(-x))]
    chain2 = [lambda x: 1/(1+np.exp(-x)), np.square]

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(16, 8))

    plot_composite_func(ax1, composite_len2_func, chain1, PLOT_RANGE)
    plot_composite_func(ax1, composite_len2_func_deriv, chain1, PLOT_RANGE)
    ax1.set_title(
        "Function and derivative of\n$f: x \\longmapsto sigmoid(x^{2})$")
    ax1.legend(["$f$", "$\\frac{df}{dx}$"])

    plot_composite_func(ax2, composite_len2_func, chain2, PLOT_RANGE)
    plot_composite_func(ax2, composite_len2_func_deriv, chain2, PLOT_RANGE)
    ax2.set_title(
        "Function and derivative of\n$f: x   \\longmapsto   [sigmoid(x)]^{2}$")
    ax2.legend(["$f$", "$\\frac{df}{dx}$"])

    plt.show()


def main():
    plot_funcs_and_derivs()


if "__main__" == __name__:
    print("Executing main")
    main()
