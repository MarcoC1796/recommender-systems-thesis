import numpy as np
from .gradientDescent import gradientDescent
from .cost_functions import get_Ju_and_DJu, get_Ja_and_DJa, get_alpha, get_beta
from .aux_functions import initializeQ, initializeP
from .accuracyMetrics import RMSE


def getRMSEs_Pfixed(
    newQs, P, R, muestral_bias=False, parameter_bias=False, alpha=None, beta=None
):
    if muestral_bias and parameter_bias:
        raise Exception("muestral_bias and parameter_bias can not be both true")
    RMSEs = []
    for i, Q_k in enumerate(newQs):
        Q_k = Q_k.reshape(R.shape[0], -1)
        if not (muestral_bias or parameter_bias):
            errors = (Q_k @ P.T - R).flatten()
        elif muestral_bias:
            eta = alpha[:, np.newaxis] + beta
            errors = (eta + Q_k @ P.T - R).flatten()
        else:
            curr_alpha = alpha[i]
            eta = curr_alpha[:, np.newaxis] + beta
            errors = (eta + Q_k @ P.T - R).flatten()
        # Check for NaN values using np.isnan()
        is_nan = np.isnan(errors)

        # Use boolean indexing to filter out NaN values
        errors = errors[~is_nan]
        RMSEs.append(RMSE(errors))

    return RMSEs


def getRMSEs_Qfixed(
    newPs, Q, R, muestral_bias=False, parameter_bias=False, alpha=None, beta=None
):
    if muestral_bias and parameter_bias:
        raise Exception("muestral_bias and parameter_bias can not be both true")
    RMSEs = []
    for i, P_k in enumerate(newPs):
        P_k = P_k.reshape(R.shape[1], -1)
        if not (muestral_bias or parameter_bias):
            errors = (Q @ P_k.T - R).flatten()
        elif muestral_bias:
            eta = alpha[:, np.newaxis] + beta
            errors = (eta + Q @ P_k.T - R).flatten()
        else:
            curr_beta = beta[i]
            eta = alpha[:, np.newaxis] + curr_beta
            errors = (eta + Q @ P_k.T - R).flatten()
        # Check for NaN values using np.isnan()
        is_nan = np.isnan(errors)

        # Use boolean indexing to filter out NaN values
        errors = errors[~is_nan]
        RMSEs.append(RMSE(errors))

    return RMSEs


def ALS(
    R,
    f,
    lambQ=0,
    lambP=0,
    muestral_bias=False,
    parameter_bias=False,
    alternations=1,
    tol=1e-3,
    max_iter=1e3,
):
    if muestral_bias and parameter_bias:
        raise Exception("muestral_bias and parameter_bias can not be both true")
    elif muestral_bias or parameter_bias:
        alpha = get_alpha(R)
        beta = get_beta(R)

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
        if muestral_bias:
            # Optimizing with respect to Q
            Ju, DJu = get_Ju_and_DJu(R, P, f, lambQ, lambP, muestral_bias=muestral_bias)
            Q = Q.flatten()
            gradientDescentResults = gradientDescent(Ju, Q, DJu, tol, max_iter)
            Q = gradientDescentResults["x_values"][-1].reshape(R.shape[0], -1)
            Ju_values = gradientDescentResults["func_values"]
            newQs = gradientDescentResults["x_values"]
            Ju_alt_values.append(Ju_values)
            RMSE_values_muestral_biased = getRMSEs_Pfixed(
                newQs, P, R, muestral_bias=True, alpha=alpha, beta=beta
            )
            RMSEu_values.append(RMSE_values_muestral_biased)

            # Optimizing with respect to P
            Ja, DJa = get_Ja_and_DJa(R, Q, f, lambQ, lambP, muestral_bias=muestral_bias)
            P = P.flatten()
            gradientDescentResults = gradientDescent(Ja, P, DJa, tol, max_iter)
            P = gradientDescentResults["x_values"][-1].reshape(R.shape[1], -1)
            Ja_values = gradientDescentResults["func_values"]
            newPs = gradientDescentResults["x_values"]
            Ja_alt_values.append(Ja_values)
            RMSE_values_muestral_biased = getRMSEs_Qfixed(
                newPs, Q, R, muestral_bias=True, alpha=alpha, beta=beta
            )
            RMSEa_values.append(RMSE_values_muestral_biased)

        elif parameter_bias:
            # Optimizing with respect to Q
            Ju, DJu = get_Ju_and_DJu(
                R, P, f, lambQ, lambP, parameter_bias=parameter_bias, beta=beta
            )
            Q = Q.flatten()
            Theta = np.append(alpha, Q)
            gradientDescentResults = gradientDescent(Ju, Theta, DJu, tol, max_iter)
            newThetas = gradientDescentResults["x_values"]
            Q = newThetas[-1][R.shape[0] :].reshape(R.shape[0], -1)
            alpha = newThetas[-1][: R.shape[0]]
            Ju_values = gradientDescentResults["func_values"]
            newQs = np.array(newThetas)[:, R.shape[0] :]
            alphas = np.array(newThetas)[:, : R.shape[0]]
            Ju_alt_values.append(Ju_values)
            RMSE_values_parameter_biased = getRMSEs_Pfixed(
                newQs, P, R, parameter_bias=True, alpha=alphas, beta=beta
            )
            RMSEu_values.append(RMSE_values_parameter_biased)

            # Optimizing with respect to P
            Ja, DJa = get_Ja_and_DJa(
                R, Q, f, lambQ, lambP, parameter_bias=parameter_bias, alpha=alpha
            )
            P = P.flatten()
            Theta = np.append(beta, P)
            gradientDescentResults = gradientDescent(Ja, Theta, DJa, tol, max_iter)
            newThetas = gradientDescentResults["x_values"]
            P = newThetas[-1][R.shape[1] :].reshape(R.shape[1], -1)
            beta = newThetas[-1][: R.shape[1]]
            Ja_values = gradientDescentResults["func_values"]
            newPs = np.array(newThetas)[:, R.shape[1] :]
            betas = np.array(newThetas)[:, : R.shape[1]]
            Ja_alt_values.append(Ja_values)
            RMSE_values_parameter_biased = getRMSEs_Qfixed(
                newPs, Q, R, parameter_bias=True, alpha=alpha, beta=betas
            )
            RMSEa_values.append(RMSE_values_parameter_biased)
        else:
            print(" Optimizing Q")
            # Optimizing with respect to Q
            Ju, DJu = get_Ju_and_DJu(R, P, f, lambQ, lambP)
            Q = Q.flatten()
            gradientDescentResults = gradientDescent(Ju, Q, DJu, tol, max_iter)
            Q = gradientDescentResults["x_values"][-1].reshape(R.shape[0], -1)
            Ju_values = gradientDescentResults["func_values"]
            newQs = gradientDescentResults["x_values"]
            Ju_alt_values.append(Ju_values)
            RMSEu_values.append(getRMSEs_Pfixed(newQs, P, R))

            print(" Oprimizing P")
            # Optimizing with respect to P
            Ja, DJa = get_Ja_and_DJa(R, Q, f, lambQ, lambP)
            P = P.flatten()
            gradientDescentResults = gradientDescent(Ja, P, DJa, tol, max_iter)
            P = gradientDescentResults["x_values"][-1].reshape(R.shape[1], -1)
            Ja_values = gradientDescentResults["func_values"]
            newPs = gradientDescentResults["x_values"]
            Ja_alt_values.append(Ja_values)
            RMSEa_values.append(getRMSEs_Qfixed(newPs, Q, R))

        print(f"Alternations Completed: {len(Ja_alt_values)}")

    return {
        "Q": Q,
        "P": P,
        "alpha": alpha if muestral_bias or parameter_bias else None,
        "beta": beta if muestral_bias or parameter_bias else None,
        "Ju_values": Ju_alt_values,
        "Ja_values": Ja_alt_values,
        "RMSEu_values": RMSEu_values,
        "RMSEa_values": RMSEa_values,
    }
