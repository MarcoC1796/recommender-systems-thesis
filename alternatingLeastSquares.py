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


def ALS(R, f, lambQ=0, lambP=0, alternations=1, tol=1e-3, max_iter=1e3):
    Q = initializeQ(R.shape[0], f)
    P = initializeP(R.shape[1], f)

    Ju_alt_values = []
    RMSEu_values = []

    Ja_alt_values = []
    RMSEa_values = []

    while len(Ja_alt_values) < alternations and (
        len(Ja_alt_values) < 2
        or np.absolute(Ja_alt_values[-1][-1] - Ja_alt_values[-2][-1])
        / np.absolute(Ja_alt_values[-2][-1])
        > tol
    ):
        # Optimizing with respect to Q
        Ju, DJu = get_Ju_and_DJu(R, P, f, lambQ, lambP)
        Q = Q.flatten()
        gradientDescentResults = gradientDescent(Ju, Q, DJu, tol, max_iter)
        Q = gradientDescentResults["x_values"][-1].reshape(R.shape[0], -1)
        Ju_values = gradientDescentResults["func_values"]
        newQs = gradientDescentResults["x_values"]
        Ju_alt_values.append(Ju_values)
        RMSEu_values.append(getRMSEs_Pfixed(newQs, P, R))

        # Optimizing with respect to P
        Ja, DJa = get_Ja_and_DJa(R, Q, f, lambQ, lambP)
        P = P.flatten()
        gradientDescentResults = gradientDescent(Ja, P, DJa, tol, max_iter)
        P = gradientDescentResults["x_values"][-1].reshape(R.shape[1], -1)
        Ja_values = gradientDescentResults["func_values"]
        newPs = gradientDescentResults["x_values"]
        Ja_alt_values.append(Ja_values)
        RMSEa_values.append(getRMSEs_Qfixed(newPs, Q, R))

    return {
        "Q": Q,
        "P": P,
        "Ju_values": Ju_alt_values,
        "Ja_values": Ja_alt_values,
        "RMSEu_values": RMSEu_values,
        "RMSEa_values": RMSEa_values,
    }
