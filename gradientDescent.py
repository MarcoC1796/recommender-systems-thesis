import numpy as np


def gradientDescent(func, initial_parameters, gradient, alpha, tol, max_iter):

    func_initial_value = func(initial_parameters)
    func_values = np.array([func_initial_value])

    parameters = initial_parameters
    i = 0

    while i < max_iter and (
        func_values.size < 2
        or np.absolute(func_values[-1] - func_values[-2]) / np.absolute(func_values[-2])
        > tol
    ):

        new_parameters = parameters - alpha * gradient(parameters)
        new_func_value = func(new_parameters)
        parameters = new_parameters
        func_values = np.append(func_values, new_func_value)
        i += 1

    return {"parameters": parameters, "func_values": func_values}
