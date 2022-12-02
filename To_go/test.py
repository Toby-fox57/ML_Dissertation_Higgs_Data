import keras.callbacks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tensorflow import keras

FILE_DIR = "SUSY.csv"
EPOCHS, BATCH = 150, 100

low = [np.arange(1, 9), "low"]  # Defines the column indexes for the low-level data.
high = [np.arange(9, 19), "high"]  # Defines the column indexes for the high-level data.
combined = [np.arange(1, 19), "comb"]  # Defines the column indexes for the combined data.

data_levels = [low, high, combined]


def principle_component_analysis(train, test, num_components):
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
    data = pd.read_csv(file_name, usecols=col_num).astype(float)
    label = pd.read_csv(file_name, usecols=[0]).astype(int)

    print(len(np.where(label==1)[0]) / len(label))

    data = keras.utils.normalize(data)

    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2)
    #y_train, y_test = keras.utils.to_categorical(y_train), keras.utils.to_categorical(y_test)

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
        model.add(keras.layers.Dense(100, activation='tanh', kernel_initializer=hidden_initializer))
    model.add(keras.layers.Dense(1, activation='sigmoid', kernel_initializer=output_initializer))

    return model


def compiler(model):
    """
    Compiles the model, a Stochastic gradient descent is used with rates given from the script, the loss is using
    a binary cross entropy as this is a binary classification and the metric of accuracy is chosen to measure the model.
    :param model: The model.
    :return: The compiled model.
    """
    sgd = keras.optimizers.SGD(learning_rate=0.005, momentum=0.9)

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
    csv_fit = keras.callbacks.CSVLogger(model_name + ".csv", separator=",", append=False)
    history = model.fit(x_train, y_train, validation_split=0.25, batch_size=batch, epochs=epochs, callbacks=[csv_fit])

    evaluation = model.evaluate(x_test, y_test)

    return history, evaluation


def print_statement(data_name, is_deep):
    """
    Prints the model use and creates the model name.
    :param data_name: Array containing the string which says whether the network is shallow or deep.
    :param is_deep: Integer of 1 or 0, 1 for deep and 0 for shallow.
    :return: String of the model name.
    """
    learning = ["shallow", "deep"]
    print("Running:", learning[is_deep], "learning for", data_name, "-level data.")

    model_name = str("History/" + learning[is_deep] + "_" + data_name)

    return model_name


def save_results(history, evaluation, model_name):
    eval_dataframe = pd.DataFrame({'epoch': ['testing'], 'accuracy': [evaluation[1]], 'loss': [evaluation[0]],
                                   'val_accuracy': [0], 'val_loss': [0]})

    results = pd.read_csv(model_name + ".csv")
    results = pd.concat([results, eval_dataframe], ignore_index=True)
    results.to_csv(model_name + ".csv", index=False)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

    ax[0].plot(history.epoch, history.history["accuracy"], label='Train')
    ax[0].plot(history.epoch, history.history["val_" + "accuracy"], linestyle="--", label='Val')
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("accuracy")
    ax[0].set_xlim(0, np.max(history.epoch))
    ax[0].legend()

    ax[1].plot(history.history['loss'], label='Train')
    ax[1].plot(history.epoch, history.history["val_" + "loss"], linestyle="--", label='Val')
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("loss")
    ax[1].set_xlim(0, np.max(history.epoch))
    ax[1].legend()

    plt.savefig(model_name + ".png")


def main(data_level, is_deep):
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

    model_name = print_statement(data_level[1], is_deep)
    model = build_model(np.shape(x_train)[1], is_deep)
    model = compiler(model)

    history, evaluation = model_fit_test(model, model_name, x_train, x_test, y_train, y_test)
    save_results(history, evaluation, model_name)

    return evaluation


def iterate():
    evaluations = np.empty((0, 2))

    for n in range(3):
        for m in range(2):
            model_evaluation = main(data_levels[n], m)
            evaluations = np.vstack((evaluations, model_evaluation))

    results = pd.read_csv("Results.csv")
    results['Accuracy'], results['Loss'] = evaluations[:, 1].tolist(), evaluations[:, 0].tolist()
    results.to_csv("Results.csv", index=False)

    return 0


iterate()