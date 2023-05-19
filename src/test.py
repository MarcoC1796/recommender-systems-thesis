import numpy as np
import matplotlib.pyplot as plt
from gradientDescent import gradientDescent
from alternatingLeastSquares import ALS
from lineSearchAlgorithm import lineSearch
from cost_functions import get_Ju_and_DJu, get_Ja_and_DJa
from aux_functions import initializeTheta, initializeX

tol = 1e-3
max_iter = 1e4

lambs = np.append(0, (np.logspace(-4, 1, 7)))

plots_per_row = 3
fig, axs = plt.subplots(
    int(np.ceil(len(lambs) / plots_per_row)),
    min(len(lambs), plots_per_row),
    figsize=(15, 10),
)
fig.suptitle(r"$Ju$ - Cost Function Regularized Plots")

for i, lamb in enumerate(lambs):
    Ju, DJu = get_Ju_and_DJu(R, P_0, f, lamb)
    Q_0 = Q_0.flatten()
    gradientDescentResults = gradientDescent(Ju, Q_0, DJu, tol, max_iter)
    Ju_values = gradientDescentResults["func_values"]
    x, y = i // plots_per_row, i % plots_per_row
    axs[x, y].plot(Ju_values)
    axs[x, y].set_title(
        r"$\lambda = {:.2e}$".format(lamb) if lamb != 0 else r"$\lambda = 0$"
    )
    if y == 0:
        axs[x, y].set(ylabel="value")
    if x == int(np.ceil(len(lambs) / plots_per_row)) - 1:
        axs[x, y].set(xlabel="iterations")

    text = "min(Ju) = {:.2e} \n\niterations = {}".format(
        Ju_values[-1], len(Ju_values) - 1
    )
    axs[x, y].text(0.2, 0.7, text, transform=axs[x, y].transAxes)

plt.show()
