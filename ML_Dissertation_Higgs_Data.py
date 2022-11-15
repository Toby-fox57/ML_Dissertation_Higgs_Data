import numpy as np
import pandas as pd
import dask.dataframe as dd
import tensorflow as tf

file_name = "HIGGS.csv"

#data = set_length(file_name)
#data = data.values.compute()

data = pd.read_csv(file_name, nrows=10000)
data = data.to_numpy()

print(np.shape(data))
print(data[0, :5])






