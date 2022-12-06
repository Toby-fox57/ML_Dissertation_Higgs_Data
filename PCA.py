import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize as normalise

FILE_DIR = "SUSY.csv.gz"

low = [np.arange(1, 9), "low-level"]  # Defines the column indexes for the low-level data.
high = [np.arange(9, 19), "high-level"]  # Defines the column indexes for the high-level data.
combined = [np.arange(1, 19), "combined"]  # Defines the column indexes for the combined data.

DATA_LEVELS = [low, high, combined]


def file_importer(col_num, file_name=FILE_DIR):
    data = pd.read_csv(file_name, nrows=100000, usecols=col_num, compression='gzip')
    data = data.to_numpy()

    data = normalise(data)

    return data


def pca_calculator(data):
    pca = PCA()
    pca.fit(data)
    pca_variance, pca_ratio = pca.explained_variance_, pca.explained_variance_ratio_

    return pca_variance, pca_ratio


def cum_ratio(data_level, pca_ratio):
    ratios = np.cumsum(pca_ratio).tolist()
    ratios = ratios + ([''] * (18 - len(data_level[0])))

    ratio_csv = pd.read_csv("PCA/PCA_ratios.csv")
    ratio_csv[data_level[1]] = ratios
    ratio_csv.to_csv("PCA/PCA_ratios.csv", index=False)


def main(data_levels=DATA_LEVELS):
    fig, axes = plt.subplots(2, 2)

    for m, col in enumerate(axes):
        for n, ax in enumerate(col):

            i = n + 2 * m

            if i != 3:
                data_level = data_levels[i]

                data = file_importer(data_level[0])
                pca_variance, pca_ratio = pca_calculator(data)

                cum_ratio(data_level, pca_ratio)

                principle_component = np.arange(1, len(data_level[0]) + 1)

                ax.bar(principle_component, pca_variance)
                ax.set_xlabel("Principal Components", fontsize=12)
                ax.set_ylabel("Variance", fontsize=12)
                ax.set_xticks(np.arange(1, len(data_level[0]) + 1))
                ax.set_xlim(0, 8.5)
                ax.set_title(data_level[1] + " features", fontsize=12)

            else:
                fig.delaxes(ax)

    fig.suptitle("Principal Component Analysis")

    plt.tight_layout()
    plt.savefig("PCA\PCA.pdf", dpi=300)
    plt.show()


main()