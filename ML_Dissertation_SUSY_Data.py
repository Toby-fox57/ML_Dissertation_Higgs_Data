import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tensorflow import keras

file_dir = "SUSY.csv"
num_rows = 100000  # Defines the amount of collision data used.

low, high = [np.arange(1, 9), "low"], [np.arange(9, 19), "high"]  # Defines the columns which have high-level and
# low-level data.
data_level, is_deep = low, 1  # Set whether to use high or low level data and if to use a deep or shallow NN.
num_components = 0  # Sets the number of principal components used, 0 = original data.


def test_and_train(data_level, num_rows, file_name):
    """
    Read the csv and stores all the data, splitting it in data and labels.
    Converts the pandas dataframe to a numpy array and then normalises the data between 0 and 1
    Splits the data into train (60%), testing (20%) and evaluation (20%).
    :param data_level: Array containing the column indexes of either low or high data
    :param num_rows: Integer of how many rows are to be read
    :param file_name: String containing the file location
    :return: Arrays containing the data and labels that are used to train, test and validation the model
    """
    data = pd.read_csv(file_name, nrows=num_rows, usecols=data_level)
    label = pd.read_csv(file_name, nrows=num_rows, usecols=[0]).astype(int)

    data, label = data.to_numpy(), label.to_numpy()
    data = keras.utils.normalize(data, axis=1)

    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.4)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)

    return x_train, x_test, x_val, y_train, y_test, y_val


def principle_component_analysis(train, test, val, num_components):
    """
    Changes the dimensions of the array from the input dimensions to the amount of principal components.
    :param train: Array of the train set with the input dimensions.
    :param test: Array of the test set with the input dimensions.
    :param val: Array of the validation set with the input dimensions.
    :param num_components: Integer of the amount of dimensions the principal component array has.
    :return: Arrays of the training, testing and validation data with dimensions equal to the principal components.
    """
    if num_components > 0:
        pca = PCA(n_components=num_components)
        pca.fit(train), pca.fit(test), pca.fit(val)

        pca_train, pca_test, pca_val = pca.transform(train), pca.transform(test), pca.transform(val)

        return pca_train, pca_test, pca_val

    else:
        return train, test, val


def build_model(input_dim, is_deep):
    """
    Builds the model, it is built with either 1 hidden layer, for a shallow network, or 5 hidden layers, for a deep
    network.
    Each layer has an initial weight which is random value from a normal distribution.
    The model is built up of,
    An input layer which has 300 units and dimensions equal to the dimensions of the data, activation of tanh is used.
    Hidden layer/s which has 300 units and an activation of tanh.
    An output layer which has 1 unit and uses an activation of a sigmoid function, as the output is either 1 or 0.
    :param input_dim: Integer of the dimensions of the data.
    :param is_deep: Integer of 1 or 0, 1 for a deep network or 0 for a shallow network.
    :return: The model.
    """
    if is_deep == 0:
        n = 1  # Shallow.

    else:
        n = 5  # Deep.

    input_initializer = keras.initializers.RandomNormal(mean=0., stddev=0.1, seed=None)
    hidden_initializer = keras.initializers.RandomNormal(mean=0., stddev=0.05, seed=None)
    output_initializer = keras.initializers.RandomNormal(mean=0., stddev=0.001, seed=None)

    model = keras.Sequential()
    model.add(keras.layers.Dense(300, input_dim=input_dim, activation='tanh', kernel_initializer=input_initializer))

    for i in range(n):
        model.add(
            keras.layers.Dense(300, activation='tanh', kernel_initializer=hidden_initializer))

    model.add(keras.layers.Dense(1, activation='sigmoid', kernel_initializer=output_initializer))

    return model


def compiler(model):
    """
    Compiles the model, a Stochastic gradient descent is used with rates given from the script, the loss is using
    a binary cross entropy as this is a binary classification and the metric of accuracy is chosen to measure the model.
    :param model: The model.
    :return: The compiled model.
    """
    sgd = keras.optimizers.SGD(learning_rate=0.05, decay=1e-5, momentum=0.9)

    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    return model


def model_fit_test(model, model_name, x_train, x_test, x_val, y_train, y_test, y_val):
    """
    The fits and evaluates the model. The fitting uses batches of 100 and fits over 1000 epochs, callbacks are used
    to implement and early stop, so the model does not over fit and to log all the training into a csv file.
    :param model: The model
    :param model_name: String determine the "model name", whether it is a shallow/deep network, using high/low level
    data and if it uses pca.
    :param x_train, x_test, x_val: Array containing the data for the training, testing and validation set.
    :param  y_train, y_test, y_val: Array containing the labels for the training, testing and validation set.
    """
    early_stop = keras.callbacks.EarlyStopping(monitor="loss", min_delta=0.00001, patience=10, verbose=1)
    csv_log = keras.callbacks.CSVLogger(model_name, separator=",", append=False)

    model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=100, epochs=1000,
              callbacks=[early_stop, csv_log])
    model.evaluate(x_test, y_test)

    return


def print_statement(data_level, is_deep, num_components):
    """
    Prints the model use and creates the model name.
    :param data_level: Array containing the string which says whether the network is shallow or deep.
    :param is_deep: Integer of 1 or 0, 1 for deep and 0 for shallow.
    :param num_components: Integer determining the number of components for the pca.
    :return: String of the model name.
    """
    learning = ["shallow", "deep"]
    print("Running:", learning[is_deep], "learning for", data_level[1], "-level data.")

    if num_components > 0:
        print("Using pca with,", num_components, "components.")
        model_name = str("History/" + learning[is_deep] + "_" + data_level[1] + "comp" + num_components + ".csv")

    else:
        model_name = str("History/" + learning[is_deep] + "_" + data_level[1] + ".csv")

    return model_name


def main(file_dir, data_level, is_deep, num_components):
    """
    Get the training, testing and validation data.
    Converts the dimensions of the data to the numver of components if pca is used.
    Prints the model name and the builds the models based of the given parameters.
    Compiles and then test the model.
    :param file_dir: String containing the location of the data
    :param data_level: Array containing the column index for the data and the name of the data used.
    :param is_deep: Integer of 1 or 0, 1 for deep and 0 for shallow.
    :param num_components: Integer determining the number of components for the pca.
    :return: Integer, 0 if the code is succesful.
    """
    x_train, x_test, x_val, y_train, y_test, y_val = test_and_train(data_level[0], num_rows, file_dir)
    x_train, x_test, x_val = principle_component_analysis(x_train, x_test, x_val, num_components)

    model_name = print_statement(data_level, is_deep, num_components)

    model = build_model(np.shape(x_train)[1], is_deep)
    model = compiler(model)
    model_fit_test(model, model_name, x_train, x_test, x_val, y_train, y_test, y_val)

    return 0


main(file_dir, data_level, is_deep, num_components)
