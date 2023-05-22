import numpy as np
from copy import deepcopy


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


def standardize_interactions(
    interactions, mean_interactions=None, std_interactions=None
):
    if mean_interactions is None:
        mean_interactions = interactions[:, 2].mean()
    if std_interactions is None:
        std_interactions = interactions[:, 2].std()
    interactions_standardized = deepcopy(interactions)
    interactions_standardized[:, 2] = (
        interactions_standardized[:, 2] - mean_interactions
    ) / standardize_interactions
    return interactions_standardized, mean_interactions, std_interactions
