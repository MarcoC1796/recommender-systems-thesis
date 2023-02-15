import numpy as np


def initializeTheta(m, p):
    return np.random.rand(m, p)


def initializeX(n, p):
    return np.random.rand(n, p)


def get_phi_and_Dphi(func, Dfunc, x_k, p_k):
    def phi(alpha):
        return func(x_k + alpha * p_k)

    def Dphi(alpha):
        return Dfunc(x_k + alpha * p_k).T @ p_k

    return phi, Dphi
