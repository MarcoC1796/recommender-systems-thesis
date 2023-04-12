import matplotlib.pyplot as plt


def plotALSResults(altValues, title):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    start = 1
    fig.suptitle(title)

    for i, values in enumerate(altValues):
        color = "b"
        label = "Ju (P fixed)"
        if i % 2 != 0:
            color = "orange"
            label = "Ja (Q fixed)"
        axs[0].plot(
            list(range(start, start + len(values))), values, color=color, label=label
        )
        axs[1].loglog(
            list(range(start, start + len(values))), values, color=color, label=label
        )

        if i < 2:
            axs[0].legend(loc="upper right")
            axs[1].legend(loc="upper right")

        start += len(values) - 1

    axs[0].set_title("Real Scale")
    axs[1].set_title("Logarithmic Scale")

    axs[0].set(xlabel="iterations", ylabel="value")
    axs[1].set(xlabel="iterations")

    plt.show()
