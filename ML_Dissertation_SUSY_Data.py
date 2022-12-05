import keras.callbacks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as area_under_curve
from tensorflow import keras

FILE_DIR = "SUSY.csv"
EPOCHS, BATCH = 150, 100

low = [np.arange(1, 9), "low"]  # Defines the column indexes for the low-level data.
high = [np.arange(9, 19), "high"]  # Defines the column indexes for the high-level data.
combined = [np.arange(1, 19), "comb"]  # Defines the column indexes for the combined data.

DATA_LEVELS = [low, high, combined]
pc = [6, 5, 9]


def principal_component_analysis(train, test, num_components):
    """
    Changes the dimensions of the array from the input dimensions to the amount of principal components.
    :param train: Array of the train set with the input dimensions.
    :param test: Array of the test set with the input dimensions.
    :param num_components: Integer of the amount of dimensions the principal component array has.
    :return: Arrays of the training, testing and validation data with dimensions equal to the principal components.
    """
    if num_components > 0:
        pca = PCA(n_components=num_components)
        pca.fit(train), pca.fit(test)

        pca_train, pca_test = pca.transform(train), pca.transform(test)

        return pca_train, pca_test

    else:
        return train, test


def test_and_train(col_num, file_name=FILE_DIR):
    """
    Read the csv and stores all the data, splitting it in data and labels.
    Converts the pandas dataframe to a numpy array and then normalises the data between 0 and 1
    Splits the data into train (80%) and testing (20%).
    :param col_num: Array containing the column indexes of either low or high data
    :param file_name: String containing the file location
    :return: Arrays containing the data and labels that are used to train and test the model
    """
    data = pd.read_csv(file_name, usecols=col_num)
    label = pd.read_csv(file_name, usecols=[0]).astype(int)

    data = keras.utils.normalize(data, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2)

    return x_train, x_test, y_train, y_test


def build_model(input_dim, is_deep):
    """
    Builds the model, it is built with either 1 hidden layer, for a shallow network, or 5 hidden layers, for a deep
    network.
    Each layer has an initial weight which is random value from a normal distribution.
    The model is built up of,
    An input layer which has 100 units and dimensions equal to the dimensions of the data, activation of tanh is used.
    Hidden layer/s which has 100 units and an activation of tanh.
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
    model.add(keras.layers.Dense(100, input_dim=input_dim, activation='tanh', kernel_initializer=input_initializer))
    for i in range(n):
        model.add(
            keras.layers.Dense(100, activation='tanh', kernel_initializer=hidden_initializer))
    model.add(keras.layers.Dense(1, activation='sigmoid', kernel_initializer=output_initializer))

    return model


def compiler(model):
    """
    Compiles the model, a Stochastic gradient descent is used with rates given from the script, the loss is using
    a binary cross entropy as this is a binary classification and the metric of accuracy is chosen to measure the model.
    :param model: The model.
    :return: The compiled model.
    """
    sgd = keras.optimizers.SGD(learning_rate=0.05, decay=1e-6, momentum=0.9)

    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    return model


def model_fit_test(model, model_name, x_train, x_test, y_train, y_test, epochs=EPOCHS, batch=BATCH):
    """
    The fits and evaluates the model. The fitting uses batches of 100 and fits over 1000 epochs
    Callbacks are used to implement and early stop, to stop over fitting and to log all the training into a csv file.
    When training 25% of the data is used to validate.
    training
    :param model: The model
    :param model_name: String determine the "model name", whether it is a shallow/deep network, using high/low level
    data and if it uses pca.
    :param x_train, x_test: Array containing the data for the training and testing set.
    :param y_train, y_test: Array containing the labels for the training and testing set.
    :param epochs: Sets the number of epochs the model will train for.
    :param batch: Sets the size of the batches.
    """
    log_dir = "History/" + model_name + ".csv"
    csv_fit = keras.callbacks.CSVLogger(log_dir, separator=",", append=False)
    history = model.fit(x_train, y_train, validation_split=0.25, batch_size=batch, epochs=epochs, callbacks=[csv_fit])

    evaluation = model.evaluate(x_test, y_test)

    return history, evaluation


def print_statement(data_name, is_deep, pc):
    """
    Prints the model use and creates the model name.
    :param data_name: Array containing the string which says whether the network is shallow or deep.
    :param is_deep: Integer of 1 or 0, 1 for deep and 0 for shallow.
    :return: String of the model name.
    """
    learning = ["shallow", "deep"]
    print("Running:", learning[is_deep], "learning for", data_name, "-level data.")

    if pc == 0:
        model_name = str(learning[is_deep] + "_" + data_name)
        
    else:
        model_name = str(learning[is_deep] + "_" + data_name + "_" + str(pc))

    return model_name


def learning_curve(history, model_name):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

    metric = ["accuracy", "loss"]

    for i, ax in enumerate(axes):
        ax.plot(history.epoch, history.history[metric[i]], label='Train')
        ax.plot(history.epoch, history.history["val_" + metric[i]], linestyle="--", label='Val')
        ax.set_xlabel("epochs", fontsize=12)
        ax.set_ylabel(metric[i], fontsize=12)
        ax.set_title("Learning curve for SUSY dataset, measure " + metric[i], fontsize=12)
        ax.set_xlim(0, np.max(history.epoch))
        ax.legend(fontsize=12)

    plt.savefig("History/" + model_name + ".pdf")


def receiver_operating_characteristic(model, model_name, x_test, y_test):
    prediction = model.predict(x_test)

    fpr, tpr, thresholds = roc_curve(y_test, prediction)
    auc = area_under_curve(fpr, tpr)

    roc_results = pd.DataFrame({"fpr": fpr, "tpr": tpr, "auc": [auc] + [''] * (len(fpr) - 1)})
    roc_results.to_csv("ROC/" + model_name + "_ROC.csv", index=False)


def model_run(data_level, is_deep, pc):
    """
    Get the training, testing and validation data.
    Converts the dimensions of the data to the number of components if pca is used.
    Prints the model name and the builds the models based of the given parameters.
    Compiles and then test the model.
    :param data_level: Array containing the column index for the data and the name of the data used.
    :param is_deep: Integer of 1 for deep or 0 for shallow.
    :return: Integer, 0 if the code is successful.
    """
    x_train, x_test, y_train, y_test = test_and_train(data_level[0])
    
    if pc != 0:
        x_train, x_test = principal_component_analysis(x_train, x_test, pc)

    model_name = print_statement(data_level[1], is_deep, pc)
    model = build_model(np.shape(x_train)[1], is_deep)
    model = compiler(model)
    
    history, evaluation = model_fit_test(model, model_name, x_train, x_test, y_train, y_test)
    receiver_operating_characteristic(model, model_name, x_test, y_test)
    learning_curve(history, model_name)

    return evaluation


def main():
    evaluations = np.empty((0, 2))

    for n in range(2):
        for m in range(3):
            model_evaluation = model_run(DATA_LEVELS[m], n, 0)
            evaluations = np.vstack((evaluations, model_evaluation))

    results = pd.read_csv("Results.csv")
    results['Accuracy'], results['Loss'] = evaluations[:, 1].tolist(), evaluations[:, 0].tolist()
    results.to_csv("Results.csv", index=False)

    return 0


def pca_main():
    evaluations = np.empty((0, 2))

    for n in range(2):
        for m in range(3):
            model_evaluation = model_run(DATA_LEVELS[m], n, pc[m])
            evaluations = np.vstack((evaluations, model_evaluation))

    results = pd.read_csv("PCA_Results.csv")
    results['Accuracy'], results['Loss'] = evaluations[:, 1].tolist(), evaluations[:, 0].tolist()
    results.to_csv("PCA_Results.csv", index=False)

    return 0


main()
# pca_main()
