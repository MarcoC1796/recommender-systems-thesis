import numpy as np


def initializeQ(m, p):
    return np.random.randn(m, p)


def initializeP(n, p):
    return np.random.randn(n, p)


def get_phi_and_Dphi(func, Dfunc, x_k, p_k):
    def phi(alpha):
        return func(x_k + alpha * p_k)

    def Dphi(alpha):
        return Dfunc(x_k + alpha * p_k).T @ p_k

    return phi, Dphi
