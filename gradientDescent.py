import numpy as np
import math
import time

from lineSearchAlgorithm import lineSearch


def gradientDescent(func, x_0, gradient, tol, max_iter=1000, alpha0=None):
    x_values = [x_0]
    func_x_0 = func(x_0)
    func_values = [func_x_0]
    alpha_values = []

    x_k = x_0
    i = 0

    while i < max_iter and (
        len(func_values) < 2
        or np.absolute(func_values[-1] - func_values[-2]) / np.absolute(func_values[-2])
        > tol
    ):
        p_k = -gradient(x_k)
        # start_time = time.time()
        alpha = lineSearch(func, gradient, x_k, p_k) if alpha0 is None else alpha0
        # end_time = time.time()
        # print(f"Time to calculate alpha: {end_time - start_time}. Alpha: {alpha}")
        x_k_next = x_k + alpha * p_k
        func_x_k_next = func(x_k_next)
        x_k = x_k_next

        x_values.append(x_k)
        func_values.append(func_x_k_next)
        alpha_values.append(alpha)

        if i % math.ceil(max_iter / 10) == 0 or i == (max_iter - 1):
            print(f"Iteration {i:4}: Cost {float(func_values[-1]):8.2f}   ")

        i += 1

    return {
        "x_values": x_values,
        "func_values": func_values,
        "alpha_values": alpha_values,
    }
