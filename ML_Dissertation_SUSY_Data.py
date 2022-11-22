import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow import keras

file_dir = "SUSY.csv"
low, high = np.arange(1, 9), np.arange(9, 19)  # Defines the columns which have high-level and low-level data.

num_rows = 100000  # Defines the number of collisions used
data_level, is_deep = low, 1

def test_and_train(data_level, num_rows, file_name):
    data = pd.read_csv(file_name, nrows=num_rows, usecols=data_level)  # Reads the data from the csv.
    label = pd.read_csv(file_name, nrows=num_rows, usecols=[0]).astype(int)  # Reads the label from the csv.

    data, label = data.to_numpy(), label.to_numpy()  # Converts from a dataframe to a numpy array.

    x_train, x_test, y_train, y_test = train_test_split(data, label,
                                                        test_size=0.1)  # splits the data into test and train.

    return x_train, x_test, y_train, y_test


def build_model(input_dim, is_deep):
    if is_deep == 0:
        n = 1

    else:
        n = 5

    input_initializer = keras.initializers.RandomNormal(mean=0., stddev=0.1)  # The weights for the input layer were
    # initialized using a random normal distribution with a mean of 0 and std of 0.1.
    hidden_initializer = keras.initializers.RandomNormal(mean=0., stddev=0.5)  # The weights for the input layer were
    # initialized using a random normal distribution with a mean of 0 and std of 0.5.
    output_initializer = keras.initializers.RandomNormal(mean=0., stddev=0.001)  # The weights for the input layer were
    # initialized using a random normal distribution with a mean of 0 and std of 0.001.

    model = keras.Sequential()
    model.add(
        keras.layers.Dense(300, input_dim=input_dim, activation='tanh', kernel_initializer=input_initializer))
    # Creates the input layer which has 300 units, the input dimension is the number of columns of data, the
    # activation is tanh ...

    for i in range(n):
        model.add(
            keras.layers.Dense(300, activation='tanh', kernel_initializer=hidden_initializer))  # Creates a hidden layer
        # containing 300 units and uses and activation of tanh ...

    model.add(
        keras.layers.Dense(1, activation='sigmoid', kernel_initializer=output_initializer))  # Output layers, uses a
    # sigmoid and contain 1 unit, this is because the output is a binary answer of signal or background.

    return model


def compiler(model):
    sgd = keras.optimizers.SGD(0.05, decay=1e-5, momentum=0.9)  # Defines the gradient descent again
    # taken from the report.

    model.compile(loss='binary_crossentropy',  # Uses binary cross entropy as the results is a binary answer.
                  optimizer=sgd,  # Uses the gradient descent as defined.
                  metrics=['accuracy'])  # Measures the model on its accuracy.

    return model


def model_fit_test(model, x_train, x_test, y_train, y_test):
    def scheduler(epoch, current_learning_rate):
        if current_learning_rate > 1e-6:
            return current_learning_rate * np.exp(-1.0000002)
        else:
            return 1e-6

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler)

    model.fit(x_train, y_train, batch_size=100, epochs=200, callbacks=[lr_schedule])
    model.evaluate(x_test, y_test, batch_size=100)

x_train, x_test, y_train, y_test = test_and_train(data_level, num_rows, file_dir)

model = build_model(np.shape(data_level)[0], is_deep)
model = compiler(model)

model_fit_test(model, x_train, x_test, y_train, y_test)
