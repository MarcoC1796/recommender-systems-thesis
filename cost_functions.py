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
        return E @ X

    return Ju, DJu


def get_Ja_and_DJa(R, Theta):
    def Ja(X):
        return np.nansum(np.square(get_E(Theta, R, X))) / 2

    def DJa(X):
        E = np.nan_to_num(get_E(Theta, R, X))
        return E.T @ Theta

    return Ja, DJa


def initializeTheta(m, p):
    return np.random.rand(m * p)


def initializeX(n, p):
    return np.random.rand(n * p)
