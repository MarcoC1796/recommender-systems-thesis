import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

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
        axs[1].semilogy(
            list(range(start, start + len(values))), values, color=color, label=label
        )

        start += len(values) - 1

        values = Pvalue
        color = "orange"
        label = "Ja (Q fixed)"
        axs[0].plot(
            list(range(start, start + len(values))), values, color=color, label=label
        )
        axs[1].semilogy(
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


def plot_hist_boxplot(data, title):
    f, (ax_box, ax_hist) = plt.subplots(
        2, sharex=True, gridspec_kw={"height_ratios": (0.15, 0.85)}
    )

    sns.boxplot(data, ax=ax_box, orient="h")
    sns.histplot(data, ax=ax_hist, discrete=True, color="steelblue", kde=True)

    # Add a vertical line at the mean value
    mean_rating = np.mean(data)
    ax_hist.axvline(x=mean_rating, color="red", linestyle="--")

    # Annotate the mean line with its value
    ax_hist.text(
        mean_rating + 0.3,
        ax_hist.get_ylim()[1] * 0.8,
        f"Mean: {mean_rating:.2f}",
        color="darkred",
        weight="bold",
    )

    # Annotate the boxplot with statistics
    median = np.median(data)
    q1 = np.percentile(data, q=25)
    q3 = np.percentile(data, q=75)

    pos_y = -0.2
    pos_median = median
    pos_q1 = q1 - (q3 - q1) / 10
    pos_q3 = q3 + (q3 - q1) / 10

    ax_box.text(
        pos_median,
        pos_y,
        f"Q2: {median:.2f}",
        verticalalignment="center",
        ha="center",
        size="medium",
        color="white",
        weight="semibold",
    )
    ax_box.text(
        pos_q1,
        pos_y,
        f"Q1: {q1:.2f}",
        verticalalignment="center",
        ha="right",
        size="medium",
        color="black",
    )
    ax_box.text(
        pos_q3,
        pos_y,
        f"Q3: {q3:.2f}",
        verticalalignment="center",
        ha="left",
        size="medium",
        color="black",
    )

    # Set the x-tick labels
    x_ticks = np.arange(-10, 11, 2)
    ax_hist.set_xticks(x_ticks)
    ax_hist.set_xticklabels(x_ticks)

    ax_box.set(xlabel="")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.suptitle(title)

    plt.show()

    # Calculate various statistics
    mean_rating = np.mean(data)
    var_rating = np.var(data, ddof=1)
    std_rating = np.std(data, ddof=1)
    median_rating = np.median(data)
    q1_rating = np.percentile(data, q=25)
    q2_rating = np.percentile(data, q=50)
    q3_rating = np.percentile(data, q=75)

    # Create a dataframe to present the statistics
    data = {
        "Metric": [
            "Mean",
            "Variance",
            "Standard deviation",
            "Median",
            "Q1",
            "Q2",
            "Q3",
        ],
        "Value": [
            mean_rating,
            var_rating,
            std_rating,
            median_rating,
            q1_rating,
            q2_rating,
            q3_rating,
        ],
    }
    stats_df = pd.DataFrame(data)

    # Format the dataframe
    stats_df["Value"] = stats_df["Value"].apply(lambda x: f"{x:.2f}")
    stats_df = stats_df.set_index("Metric")

    return stats_df
