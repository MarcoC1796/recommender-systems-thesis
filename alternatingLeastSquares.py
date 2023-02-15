import numpy as np
from gradientDescent import gradientDescent
from cost_functions import get_Ju_and_DJu, get_Ja_and_DJa
from aux_functions import initializeTheta, initializeX


def ALS(R, p, alpha, als_tol, gd_tol, als_max_iter, gd_max_iter):

    Theta = initializeTheta(R.shape[0], p)

    X = initializeX(R.shape[1], p)

    J_values = np.array([])

    i = 0

    while i < als_max_iter:

        Ju, DJu = get_Ju_and_DJu(R, X, p)
        Theta = Theta.flatten()
        newThetaResult = gradientDescent(Ju, Theta, DJu, gd_tol, gd_max_iter)
        Theta = newThetaResult["parameters"].reshape(R.shape[0], p)
        Ju_values = newThetaResult["func_values"]
        J_values = np.append(J_values, Ju_values)

        Ja, DJa = get_Ja_and_DJa(R, Theta, p)
        X = X.flatten()
        newXResult = gradientDescent(Ja, X, DJa, gd_tol, gd_max_iter)
        X = newXResult["parameters"].reshape(R.shape[1], p)
        Ja_values = newXResult["func_values"]
        J_values = np.append(J_values, Ja_values)

        i += 1

    return {"Theta": Theta, "X": X, "J_values": J_values}
