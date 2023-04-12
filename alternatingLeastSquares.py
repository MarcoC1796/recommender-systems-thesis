import numpy as np
from gradientDescent import gradientDescent
from cost_functions import get_Ju_and_DJu, get_Ja_and_DJa
from aux_functions import initializeQ, initializeP
from accuracyMetrics import RMSE


def getRMSEs_Pfixed(newQs, P, R):
    RMSEs = []
    for Q_k in newQs:
        Q_k = Q_k.reshape(R.shape[0], -1)
        errors = (Q_k @ P.T - R).flatten()

        # Check for NaN values using np.isnan()
        is_nan = np.isnan(errors)

        # Use boolean indexing to filter out NaN values
        errors = errors[~is_nan]
        RMSEs.append(RMSE(errors))

    return RMSEs


def getRMSEs_Qfixed(newPs, Q, R):
    RMSEs = []
    for P_k in newPs:
        P_k = P_k.reshape(R.shape[1], -1)
        errors = (Q @ P_k.T - R).flatten()

        # Check for NaN values using np.isnan()
        is_nan = np.isnan(errors)

        # Use boolean indexing to filter out NaN values
        errors = errors[~is_nan]
        RMSEs.append(RMSE(errors))

    return RMSEs


def ALS(R, f, alternations, tol, max_iter):
    Q = initializeQ(R.shape[0], f)
    P = initializeP(R.shape[1], f)

    RMSEs = []
    J_values = []

    i = 0

    while i < alternations:
        # Optimizing with respect to Q
        Ju, DJu = get_Ju_and_DJu(R, P, f)
        Q = Q.flatten()
        gradientDescentResults = gradientDescent(Ju, Q, DJu, tol, max_iter)
        Q = gradientDescentResults["x_values"][-1].reshape(R.shape[0], -1)
        Ju_values = gradientDescentResults["func_values"]
        newQs = gradientDescentResults["x_values"]
        J_values.append(Ju_values)
        RMSEs.append(getRMSEs_Pfixed(newQs, P, R))

        # Optimizing with respect to P
        Ja, DJa = get_Ja_and_DJa(R, Q, f)
        P = P.flatten()
        gradientDescentResults = gradientDescent(Ja, P, DJa, tol, max_iter)
        P = gradientDescentResults["x_values"][-1].reshape(R.shape[1], -1)
        Ja_values = gradientDescentResults["func_values"]
        newPs = gradientDescentResults["x_values"]
        J_values.append(Ja_values)
        RMSEs.append(getRMSEs_Qfixed(newPs, Q, R))

        i += 1

    return {"Q": Q, "P": P, "J_values": J_values, "RMSEs": RMSEs}
