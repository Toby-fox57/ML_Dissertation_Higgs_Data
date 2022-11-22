import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow import keras

file_dir = "SUSY.csv"
low, high = np.arange(1, 9), np.arange(9, 19)  # Defines the columns which have high-level and low-level data.

num_rows = 100000  # Defines the amount of collision data used.
data_level, is_deep = high, 0  # Set whether to use high or low level data and if to use a deep or shallow NN.


def test_and_train(data_level, num_rows, file_name):
    data = pd.read_csv(file_name, nrows=num_rows, usecols=data_level)  # Reads the data from the csv.
    label = pd.read_csv(file_name, nrows=num_rows, usecols=[0]).astype(int)  # Reads the label from the csv.

    data, label = data.to_numpy(), label.to_numpy()  # Converts from a dataframe to a numpy array.

    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.1)  # splits the data into test
    # and train, with 90% of the data used to train and 10% used to test.

    return x_train, x_test, y_train, y_test


def build_model(input_dim, is_deep):
    if is_deep == 0:
        n = 1  # Sets the number of hidden layers to 1 if it is a shallow NN.

    else:
        n = 5  # Sets the number of hidden layers to 5 if it is a deep NN.

    input_initializer = keras.initializers.RandomNormal(mean=0., stddev=0.1)  # The weights for the input layer were
    # initialized using a random normal distribution with a mean of 0 and std of 0.1.
    hidden_initializer = keras.initializers.RandomNormal(mean=0., stddev=0.05)  # The weights for the input layer were
    # initialized using a random normal distribution with a mean of 0 and std of 0.05.
    output_initializer = keras.initializers.RandomNormal(mean=0., stddev=0.001)  # The weights for the input layer were
    # initialized using a random normal distribution with a mean of 0 and std of 0.001.

    model = keras.Sequential()
    model.add(
        keras.layers.Dense(300, input_dim=input_dim, activation='tanh', kernel_initializer=input_initializer))
    # Creates the input layer which has 300 units, the input dimension is the number of columns of data
    # An activation function of tanh is used as it is zero centred meaning it will have a strong gradient which works
    # well will a binary result.

    for i in range(n):
        model.add(
            keras.layers.Dense(300, activation='tanh', kernel_initializer=hidden_initializer))  # Creates a hidden layer
        # containing 300 units and uses an activation function of tanh.

    model.add(
        keras.layers.Dense(1, activation='sigmoid', kernel_initializer=output_initializer))  # Output layer, has 1 unit
    #  as the result is binary answer, uses a sigmoid function as the answer is 0 or 1, so it cannot be zero centred.

    return model


def compiler(model):
    sgd = keras.optimizers.SGD(lr=0.05, decay=1e-5, momentum=0.9)  # Defines the gradient descent again
    # taken from the report.

    model.compile(loss='binary_crossentropy',  # Uses binary cross entropy as the results is a binary answer.
                  optimizer=sgd,  # Uses the gradient descent as defined.
                  metrics=['accuracy'])  # Measures the model on its accuracy.

    return model


def scheduler(epoch, current_learning_rate):
    if current_learning_rate > 1e-6:   # Define a learing rate schedule, such that the learning rate decays by
        return current_learning_rate / 1.0000002  # a factor of 1.0000002 until the learning rate reaches 1e-6.

    else:
        return 1e-6


def model_fit_test(model, x_train, x_test, y_train, y_test):
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler)
    early_stop = keras.callbacks.EarlyStopping(monitor="loss", min_delta=0.00001, patience=10, verbose=1)  # Steps
    # criteria to stop early, when the change in the loss over 10 epochs is less than 0.00001 the code will stop early
    # the is done to stop over-fitting.

    model.fit(x_train, y_train, batch_size=100, epochs=1000, callbacks=[lr_schedule, early_stop])  # fits the model
    # using the training data.
    model.evaluate(x_test, y_test, batch_size=100)  # evaluates the trained model against the test data.


x_train, x_test, y_train, y_test = test_and_train(data_level, num_rows, file_dir)

model = build_model(np.shape(data_level)[0], is_deep)
model = compiler(model)

model_fit_test(model, x_train, x_test, y_train, y_test)
