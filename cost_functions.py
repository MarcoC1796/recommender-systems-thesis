import numpy as np


def get_alpha(R):
    return np.nansum(R, 1) / np.count_nonzero(~np.isnan(R), 1)


def get_beta(R):
    return np.nansum(R, 0) / np.count_nonzero(~np.isnan(R), 0)


def get_muestral_biases(R):
    alpha = get_alpha(R)
    beta = get_beta(R)
    eta = alpha[:, np.newaxis] + beta
    return eta


def get_Ju_and_DJu(
    R, P, f, lambQ=0, lambP=0, muestral_bias=False, parameter_bias=False, beta=None
):
    """
    Arguments:
        R: m P n matriP.
        P: n P p matriP
    """
    if muestral_bias and parameter_bias:
        raise Exception("muestral_bias and parameter_bias can not be both true")
    elif parameter_bias and beta is None:
        raise Exception("if parameter_bias is true, beta must be specified")

    m, _ = R.shape

    if muestral_bias:
        eta = get_muestral_biases(R)

        def Ju(Q):
            """
            Arguments:
                Q: a flattened array representing a m x p matrix
            """
            Q = Q.reshape((m, f))
            return (
                np.nansum(np.square(eta + Q @ P.T - R)) / 2
                + (lambQ / 2) * np.nansum(np.square(Q))
                + (lambP / 2) * np.nansum(np.square(P))
            )

        def DJu(Q):
            """
            Arguments:
                Q: a flattened array representing a m x p matrix
            """
            Q = Q.reshape((m, f))
            E = np.nan_to_num(eta + Q @ P.T - R)
            DJu_values = E @ P + lambQ * Q
            return DJu_values.flatten()

    elif parameter_bias:

        def Ju(Theta):
            """
            Arguments:
                Q: a flattened array representing a m x p matrix
            """
            alpha = Theta[:m]
            Q = Theta[m:]
            Q = Q.reshape((m, f))
            eta = alpha[:, np.newaxis] + beta

            return (
                np.nansum(np.square(eta + Q @ P.T - R)) / 2
                + (lambQ / 2) * np.nansum(np.square(Q))
                + (lambP / 2) * np.nansum(np.square(P))
            )

        def DJu(Theta):
            """
            Arguments:
                Q: a flattened array representing a m x p matrix
            """
            alpha = Theta[:m]
            Q = Theta[m:]
            Q = Q.reshape((m, f))

            eta = alpha[:, np.newaxis] + beta
            E = np.nan_to_num(eta + Q @ P.T - R)
            DJu_DQ = E @ P + lambQ * Q
            DJu_Dalpha = np.nansum(E, 1)

            DJu_values = np.append(DJu_Dalpha.flatten(), DJu_DQ)

            return DJu_values

    else:

        def Ju(Q):
            """
            Arguments:
                Q: a flattened array representing a m x p matrix
            """
            Q = Q.reshape((m, f))

            return (
                np.nansum(np.square(Q @ P.T - R)) / 2
                + (lambQ / 2) * np.nansum(np.square(Q))
                + (lambP / 2) * np.nansum(np.square(P))
            )

        def DJu(Q):
            """
            Arguments:
                Q: a flattened array representing a m x p matrix
            """
            Q = Q.reshape((m, f))
            E = np.nan_to_num(Q @ P.T - R)
            DJu_values = E @ P + lambQ * Q
            return DJu_values.flatten()

    return Ju, DJu


def get_Ja_and_DJa(
    R, Q, f, lambQ=0, lambP=0, muestral_bias=False, parameter_bias=False, alpha=None
):
    if muestral_bias and parameter_bias:
        raise Exception("muestral_bias and parameter_bias can not be both true")
    elif parameter_bias and alpha is None:
        raise Exception("if parameter_bias is true, beta must be specified")
    _, n = R.shape

    if muestral_bias:
        eta = get_muestral_biases(R)

        def Ja(P):
            P = P.reshape((n, f))
            return (
                np.nansum(np.square(eta + Q @ P.T - R)) / 2
                + (lambQ / 2) * np.nansum(np.square(Q))
                + (lambP / 2) * np.nansum(np.square(P))
            )

        def DJa(P):
            P = P.reshape((n, f))
            E = np.nan_to_num(eta + Q @ P.T - R)
            DJa_values = E.T @ Q + lambP * P
            return DJa_values.flatten()

    elif parameter_bias:

        def Ja(Theta):
            beta = Theta[:n]
            P = Theta[n:]
            P = P.reshape((n, f))
            eta = alpha[:, np.newaxis] + beta

            return (
                np.nansum(np.square(eta + Q @ P.T - R)) / 2
                + (lambQ / 2) * np.nansum(np.square(Q))
                + (lambP / 2) * np.nansum(np.square(P))
            )

        def DJa(Theta):
            beta = Theta[:n]
            P = Theta[n:]
            P = P.reshape((n, f))

            eta = alpha[:, np.newaxis] + beta
            E = np.nan_to_num(eta + Q @ P.T - R)
            DJa_DQ = E.T @ Q + lambP * P
            DJa_Dalpha = np.nansum(E, 0)

            DJa_values = np.append(DJa_Dalpha.flatten(), DJa_DQ)

            return DJa_values.flatten()

    else:

        def Ja(P):
            P = P.reshape((n, f))
            return (
                np.nansum(np.square(Q @ P.T - R)) / 2
                + (lambQ / 2) * np.nansum(np.square(Q))
                + (lambP / 2) * np.nansum(np.square(P))
            )

        def DJa(P):
            P = P.reshape((n, f))
            E = np.nan_to_num(Q @ P.T - R)
            DJa_values = E.T @ Q + lambP * P
            return DJa_values.flatten()

    return Ja, DJa
