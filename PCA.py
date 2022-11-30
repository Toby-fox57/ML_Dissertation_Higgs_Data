import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from tensorflow import keras

file_dir = "SUSY.csv"

low = [np.arange(1, 9), "low"]  # Defines the column indexes for the low-level data.
high = [np.arange(9, 19), "high"]  # Defines the column indexes for the high-level data.
combined = [np.arange(1, 19), "comb"]  # Defines the column indexes for the combined data.

data_levels = [low, high, combined]


def file_importer(col_num, file_name):
    data = pd.read_csv(file_name, usecols=col_num).astype(float)
    data = data.to_numpy()

    data = keras.utils.normalize(data)

    return data


def pca_calculator(data):
    pca = PCA()
    pca.fit(data)
    pca_data = pca.transform(data)

    return pca_data


def plotter(data_level, data):
    variance, principle_component = np.var(data, 0), data_level[0]

    plt.bar(principle_component, variance)
    plt.xlabel("Principal Components")
    plt.ylabel("Variance")

    plt.savefig("PCA\PCA_" + data_level[1] + ".png")
    plt.show()


def main(data_level):
    data = file_importer(data_level[0], file_dir)
    pca_data = pca_calculator(data)
    plotter(data_level, pca_data)


for level in data_levels:
    main(level)
