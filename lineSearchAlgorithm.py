def lineSearchAlgorithm(alpha_max=128):
    if alpha_max < 0:
        raise Exception("alpha_max must be positive")

    alpha_i_1 = 0
    alpha_i = (alpha_max - alpha_i_1) / 2
