import numpy as np


def get_Ju_and_DJu(R, P, f, lambQ=0, lambP=0):
    """
    Arguments:
        R: m P n matriP.
        P: n P p matriP
    """

    m, _ = R.shape

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


def get_Ja_and_DJa(R, Q, f, lambQ, lambP):
    _, n = R.shape

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
