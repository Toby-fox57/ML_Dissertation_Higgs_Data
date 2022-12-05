import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize as normalise

file_dir = "SUSY.csv"

low = [np.arange(1, 9), "low"]  # Defines the column indexes for the low-level data.
high = [np.arange(9, 19), "high"]  # Defines the column indexes for the high-level data.
combined = [np.arange(1, 19), "combined"]  # Defines the column indexes for the combined data.

data_levels = [low, high, combined]


def file_importer(col_num, file_name):
    data = pd.read_csv(file_name, usecols=col_num).astype(float)
    data = data.to_numpy()

    data = normalise(data)

    return data


def pca_calculator(data):
    pca = PCA()
    pca.fit(data)
    pca_variance, pca_ratio = pca.explained_variance_, pca.explained_variance_ratio_

    return pca_variance, pca_ratio


def plotter(data_level, pca_variance):
    principle_component = np.arange(1, len(data_level[0]) + 1)

    plt.bar(principle_component, pca_variance)
    plt.xlabel("Principal Components", fontsize=12)
    plt.ylabel("Variance", fontsize=12)
    plt.title("Principal Component Analysis of " + data_level[1] + "-level features", fontsize=12)
    plt.xticks(np.arange(1, len(data_level[0]) + 1))

    plt.savefig("PCA\PCA_" + data_level[1] + ".pdf")
    plt.show()


def cum_ratio(data_level, pca_ratio):
    principle_component, ratios = np.arange(1, len(data_level[0]) + 1), np.cumsum(pca_ratio)

    ratios = pd.DataFrame({'Principal Component': principle_component, 'Variance Ratio': ratios})
    ratios.to_csv("PCA/"+data_level[1]+"_ratios.csv", index=False)


def main(data_level):
    data = file_importer(data_level[0], file_dir)
    pca_variance, pca_ratio = pca_calculator(data)
    plotter(data_level, pca_variance)
    cum_ratio(data_level, pca_ratio)


for level in data_levels:
    main(level)
