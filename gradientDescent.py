import numpy as np

from lineSearchAlgorithm import lineSearch


def gradientDescent(func, x_0, gradient, tol, max_iter=1000, alpha=1e-4):

    x_values = [x_0]
    func_x_0 = func(x_0)
    func_values = np.array([func_x_0])
    alpha_values = np.array([])

    x_k = x_0
    i = 0

    while i < max_iter and (
        func_values.size < 2
        or np.absolute(func_values[-1] - func_values[-2]) / np.absolute(func_values[-2])
        > tol
    ):

        p_k = -gradient(x_k)
        alpha = lineSearch(func, gradient, x_k, p_k)
        x_k_next = x_k + alpha * p_k
        func_x_k_next = func(x_k_next)
        x_k = x_k_next

        x_values.append(x_k)
        func_values = np.append(func_values, func_x_k_next)
        alpha_values = np.append(alpha_values, alpha)
        i += 1

    return {
        "x_values": x_values,
        "func_values": func_values,
        "alpha_values": alpha_values,
    }
