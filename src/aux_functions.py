import numpy as np


def initializeQ(m, f):
    return np.random.randn(m, f)


def initializeP(n, f):
    return np.random.randn(n, f)


def get_phi_and_Dphi(func, Dfunc, x_k, p_k):
    def phi(alpha):
        return func(x_k + alpha * p_k)

    def Dphi(alpha):
        return Dfunc(x_k + alpha * p_k).T @ p_k

    return phi, Dphi
