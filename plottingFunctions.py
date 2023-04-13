import matplotlib.pyplot as plt


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
