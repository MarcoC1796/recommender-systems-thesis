import numpy as np


def get_Ju_and_DJu(R, X, p):
    """
    Arguments:
        R: m x n matrix.
        X: n x p matrix
    """

    m, _ = R.shape

    def Ju(Theta):
        """
        Arguments:
            Theta: a flattened array representing a m x p matrix
        """
        Theta = Theta.reshape((m, p))
        return np.nansum(np.square(Theta @ X.T - R)) / 2

    def DJu(Theta):
        """
        Arguments:
            Theta: a flattened array representing a m x p matrix
        """
        Theta = Theta.reshape((m, p))
        E = np.nan_to_num(Theta @ X.T - R)
        DJu_values = E @ X
        return DJu_values.flatten()

    return Ju, DJu


def get_Ja_and_DJa(R, Theta, p):
    _, n = R.shape

    def Ja(X):
        X = X.reshape((n, p))
        return np.nansum(np.square(Theta @ X.T - R)) / 2

    def DJa(X):
        X = X.reshape((n, p))
        E = np.nan_to_num(Theta @ X.T - R)
        DJa_values = E.T @ Theta
        return DJa_values.flatten()

    return Ja, DJa
