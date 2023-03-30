import numpy as np
from gradientDescent import gradientDescent
from cost_functions import get_Ju_and_DJu, get_Ja_and_DJa
from aux_functions import initializeQ, initializeP


def ALS(R, f, alpha, als_tol, gd_tol, als_max_iter, gd_max_iter):

    Q = initializeQ(R.shape[0], f)
    newQs = [Q]

    P = initializeP(R.shape[1], f)
    newPs = [P]

    errors = []

    J_values = np.array([])

    i = 0

    while i < als_max_iter:

        Ju, DJu = get_Ju_and_DJu(R, P, f)
        Q = Q.flatten()
        gradientDescentResults = gradientDescent(Ju, Q, DJu, gd_tol, gd_max_iter)
        Q = gradientDescentResults["x_values"][-1].reshape(R.shape[0], -1)
        Ju_values = gradientDescentResults["func_values"]
        J_values = np.append(J_values, Ju_values)

        Ja, DJa = get_Ja_and_DJa(R, Q, f)
        P = P.flatten()
        gradientDescentResults = gradientDescent(Ja, P, DJa, gd_tol, gd_max_iter)
        P = gradientDescentResults["x_values"][-1].reshape(R.shape[1], -1)
        Ja_values = gradientDescentResults["func_values"]
        J_values = np.append(J_values, Ja_values)

        i += 1

    return {"Q": Q, "P": P, "J_values": J_values, "errors": errors}
