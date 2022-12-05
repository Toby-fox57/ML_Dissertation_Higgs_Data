import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILE_DIR = "SUSY.csv"

low = [[1, 4, 7, 8], "low"]  # Defines the column indexes for the low-level data.
high = [np.arange(9, 19), "high"]  # Defines the column indexes for the high-level data.
combined = [np.arange(1, 19), "comb"]  # Defines the column indexes for the combined data.

DATA_LEVELS = [low, high, combined]


def file_importer(col_num, file_name=FILE_DIR):
    label = pd.read_csv(file_name, usecols=[0]).astype(float)
    data = pd.read_csv(file_name, usecols=col_num).astype(float)

    return label, data


def low_feature_analysis(data_level=DATA_LEVELS[0]):
    label, data = file_importer(data_level[0])

    feature_names = ["Lepton 1 $p_T$ (GeV)", "Lepton 2 $p_T$ (GeV)", "$E_{mis}$ (GeV)", ""]
    label, data = label.to_numpy(), data.to_numpy()*100

    signal, background = np.where(label == 1)[0], np.where(label == 0)[0]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.subplots_adjust(wspace=0.3)

    for i, col in enumerate(axes):
        for m, ax in enumerate(col):

            n = i*2 + m

            ax.hist(data[signal, n], bins=100, histtype='step', label='signal', color='r',
                    weights=np.zeros_like(data[signal, n]) + 1. / data[signal, n].size)
            ax.hist(data[background, n], bins=100, histtype='step', label='background', color='k',
                    weights=np.zeros_like(data[background, n]) + 1. / data[background, n].size)
            ax.set_ylabel("Fraction of events", fontsize=12)
            ax.set_xlabel(str(feature_names[n]), fontsize=12)

            Line, Label = ax.get_legend_handles_labels()

            ax.set_xlim(0, 200)
            if n != 1:
                ax.set_ylim(0, 0.2)
                ax.set_yticks(np.linspace(0, 0.2, 5))
            if n == 3:
                fig.delaxes(ax)

    fig.suptitle("Diustribution of "+ data_level[1] + " level features of the SUSY data set", fontsize=12)
    fig.legend(Line, Label, loc='lower right', fontsize=12)

    plt.savefig("Features/" + data_level[1] + "_level.pdf")
    plt.show()

    return 0


def high_feature_analysis(data_level=DATA_LEVELS[1]):
    label, data = file_importer(data_level[0])

    feature_names = ["Mis. $E_T^{rel}$", "Axial Mis. $E_T$", "$M_R$", r"$M_T^R$", "R", "$M_{T2}$", "$S_R$",
                     r"$M_{\Delta}^R$", r"$\Delta\phi^{\beta}_R$", r"$cos(\theta_{R+1})$"]
    label, data = label.to_numpy(), data.to_numpy()*100
    data[:, 4], data[:, 8:10] = data[:, 4] / 100, data[:, 8:10] / 100

    signal, background = np.where(label == 1)[0], np.where(label == 0)[0]

    fig, axes = plt.subplots(4, 3, figsize=(12, 9))
    fig.subplots_adjust(wspace=0.3, hspace=0.5)


    for i, col in enumerate(axes):
        for m, ax in enumerate(col):

            n = i*3 + m

            if n == 10:
                n = 9

                ax.hist(data[signal, n], bins=50, histtype='step', label='signal', color='r',
                              weights=np.zeros_like(data[signal, n]) + 1. / data[signal, n].size)
                ax.hist(data[background, n], bins=50, histtype='step', label='background', color='k',
                              weights=np.zeros_like(data[background, n]) + 1. / data[background, n].size)
                ax.set_ylabel("Fraction of events", fontsize=12)
                ax.set_xlabel(str(feature_names[n]), fontsize=12)

                ax.set_ylim(0, 0.1)
                ax.set_xlim(0, 1)


            elif n != 9 and n != 11:
                ax.hist(data[signal, n], bins=50, histtype='step', label='signal', color='r',
                              weights=np.zeros_like(data[signal, n]) + 1. / data[signal, n].size)
                ax.hist(data[background, n], bins=50, histtype='step', label='background', color='k',
                              weights=np.zeros_like(data[background, n]) + 1. / data[background, n].size)
                ax.set_ylabel("Fraction of events", fontsize=12)
                ax.set_xlabel(str(feature_names[n]), fontsize=12)

                Line, Label = ax.get_legend_handles_labels()

                if n == 0:
                    ax.set_xlim(0, 300)
                    ax.set_yticks(np.linspace(0, 0.2, 5))

                elif n == 1:
                    ax.set_xlim(-300, 300)
                    ax.set_yticks(np.linspace(0, 0.2, 5))

                elif n == 2:
                    ax.set_xlim(0, 200)

                elif n == 3:
                    ax.set_yticks(np.linspace(0, 0.2, 5))
                    ax.set_xlim(0, 200)

                elif n == 4:
                    ax.set_xlim(0, 1)
                    ax.set_yticks(np.linspace(0, 0.1, 5))

                elif n == 5:
                    ax.set_xlim(0, 200)

                elif n == 6:
                    ax.set_xlim(0, 250)

                elif n == 7:
                    ax.set_xlim(0, 250)
                    ax.set_yticks(np.linspace(0, 0.2, 5))

                elif n == 8:
                    ax.set_xlim(0, 1.5)

            else:
                fig.delaxes(ax)

    fig.suptitle("Distribution of "+ data_level[1] + " level features of the SUSY data set", fontsize=12)
    fig.legend(Line, Label, loc='lower right', fontsize=12)

    plt.savefig("Features/" + data_level[1] + "_level.pdf")
    plt.show()

    return 0

def main():
    low_feature_analysis()
    high_feature_analysis()

    return 0


main()
