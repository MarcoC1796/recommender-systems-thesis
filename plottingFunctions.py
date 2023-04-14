import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from aux_functions import initializeQ, initializeP
from cost_functions import get_Ja_and_DJa, get_Ju_and_DJu
from gradientDescent import gradientDescent


def plotALSResults(Qvalues, Pvalues, title, altStart=0):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    start = 0
    fig.suptitle(title)

    for i, (Qvalue, Pvalue) in enumerate(
        zip(Qvalues[altStart:], Pvalues[altStart:]), altStart
    ):
        values = Qvalue
        color = "b"
        label = "Ju (P fixed)"
        axs[0].plot(
            list(range(start, start + len(values))), values, color=color, label=label
        )
        axs[1].loglog(
            list(range(start, start + len(values))), values, color=color, label=label
        )

        start += len(values) - 1

        values = Pvalue
        color = "orange"
        label = "Ja (Q fixed)"
        axs[0].plot(
            list(range(start, start + len(values))), values, color=color, label=label
        )
        axs[1].loglog(
            list(range(start, start + len(values))), values, color=color, label=label
        )

        start += len(values) - 1

        if i < 1:
            axs[0].legend(loc="upper right")
            axs[1].legend(loc="upper right")

    axs[0].set_title("Real Scale")
    axs[1].set_title("Logarithmic Scale")

    axs[0].set(xlabel="iterations", ylabel="value")
    axs[1].set(xlabel="iterations")

    plt.show()


def plot_Ju_Ja_as_functions_of_lambda(R, f):
    np.random.seed(43)
    Q_0 = initializeQ(R.shape[0], f)
    P_0 = initializeP(R.shape[1], f)

    lambs = np.append(0, (np.logspace(0, 4, 100)))
    Ju_values = []
    Ja_values = []

    for lamb in lambs:
        Ju, _ = get_Ju_and_DJu(R, P_0, f, lamb)
        Ja, _ = get_Ja_and_DJa(R, Q_0, f, lamb)
        Ju_values.append(Ju(Q_0.flatten()))
        Ja_values.append(Ja(P_0.flatten()))

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(r"Cost Functions as functions of $\lambda$")

    ax[0].plot(lambs, Ju_values, color=sns.color_palette()[0])
    ax[0].set(ylabel="value", xlabel=r"$\lambda$")
    ax[0].set_title(r"$Ju(\lambda)$")

    ax[1].plot(lambs, Ja_values, color=sns.color_palette()[1])
    ax[1].set(xlabel=r"$\lambda$")
    ax[1].set_title(r"$Ja(\lambda)$")

    plt.show()


def plot_Ju_Ja_fixed_lamb(R, lamb, f, max_iter=1e3, tol=1e-3, seed=43):
    np.random.seed(43)
    Q_0 = initializeQ(R.shape[0], f)
    P_0 = initializeP(R.shape[1], f)

    Ju, DJu = get_Ju_and_DJu(R, P_0, f, lamb)
    Ja, DJa = get_Ja_and_DJa(R, Q_0, f, lamb)

    gradientDescentResults = gradientDescent(Ju, Q_0.flatten(), DJu, tol, max_iter)
    Ju_values = gradientDescentResults["func_values"]
    gradientDescentResults = gradientDescent(Ja, P_0.flatten(), DJa, tol, max_iter)
    Ja_values = gradientDescentResults["func_values"]

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(r"Regularized cost functions with common $\lambda={0}$".format(lamb))

    ax[0].plot(Ju_values, color=sns.color_palette()[0], label=r"$Ju(\lambda)$")
    ax[0].set(ylabel="value", xlabel=r"iterations")
    ax[0].legend()

    ax[1].plot(Ja_values, color=sns.color_palette()[1], label=r"$Ja(\lambda)$")
    ax[1].set(xlabel=r"iterations")
    ax[1].legend()

    plt.show()


def plot_costFunction_with_different_lamb(
    R, f, X_0, Theta, getCostFunc, lambs, title, max_iter=1e3, tol=1e-3, logScale=False
):
    plots_per_row = 3

    fig, axs = plt.subplots(
        int(np.ceil(len(lambs) / plots_per_row)),
        min(len(lambs), plots_per_row),
        figsize=(15, 10),
    )
    fig.suptitle(title)

    for i, lamb in enumerate(lambs):
        J, DJ = getCostFunc(R, Theta, f, lamb)
        X_0 = X_0.flatten()
        gradientDescentResults = gradientDescent(J, X_0, DJ, tol, max_iter)
        J_values = gradientDescentResults["func_values"]
        x, y = i // plots_per_row, i % plots_per_row
        if not logScale:
            axs[x, y].plot(J_values, color=sns.color_palette()[i])
        else:
            axs[x, y].loglog(J_values, color=sns.color_palette()[i])
        axs[x, y].set_title(
            r"$\lambda = {:.1e}$".format(lamb) if lamb != 0 else r"$\lambda = 0$"
        )
        if y == 0:
            axs[x, y].set(ylabel="value")
        if x == int(np.ceil(len(lambs) / plots_per_row)) - 1:
            axs[x, y].set(xlabel="iterations")

        text = "min(Cost) = {:.2e} \n\niterations = {}".format(
            J_values[-1], len(J_values) - 1
        )
        axs[x, y].text(0.2, 0.7, text, transform=axs[x, y].transAxes)

    plt.show()
