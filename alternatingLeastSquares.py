import numpy as np
from gradientDescent import gradientDescent
from cost_functions import *


def ALS(R, p, alpha, als_tol, gd_tol, als_max_iter, gd_max_iter):

    Theta = initializeTheta(R.shape[0], p)
    X = initializeX(R.shape[1], p)
    J_values = np.array([])

    Ju, DJu = get_Ju_and_DJu(R, X)

    Ja, DJa = get_Ja_and_DJa(R, Theta)

    i = 0

    while i < als_max_iter:

        Ju, DJu = get_Ju_and_DJu(R, X)
        newThetaResult = gradientDescent(Ju, Theta, DJu, alpha, gd_tol, gd_max_iter)
        Theta = newThetaResult["parameters"]
        Ju_values = newThetaResult["func_values"]
        J_values = np.append(J_values, Ju_values)

        Ja, DJa = get_Ja_and_DJa(R, Theta)
        newXResult = gradientDescent(Ja, X, DJa, alpha, gd_tol, gd_max_iter)
        X = newXResult["parameters"]
        Ja_values = newXResult["func_values"]
        J_values = np.append(J_values, Ja_values)

        i += 1

    return {"Theta": Theta, "X": X, "J_values": J_values}
