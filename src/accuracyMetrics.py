import numpy as np


def RMSE(errors):
    return np.sqrt(np.sum(np.square(errors)) / errors.size)
