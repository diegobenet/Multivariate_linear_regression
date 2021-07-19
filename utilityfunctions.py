""" utilityfunctions.py
    This python script has the methods needed to implement the multivariate linear regression.

    Authors:
        Daniel Alberto Martínez Sánchez     534032
        Diego Elizondo Benet                567003
        Alejandro Flores Ramones            537489
    Emails:
        daniel.martinezs@udem.edu
        diego.elizondob@udem.edu
        alejandro.floresr@udem.edu

    Institution: Universidad de Monterrey
    First created: Wednesday 04 Nov 2020

    We hereby declare that we've worked on this activity with academic integrity.
"""

# import standard libraries
import numpy as np
import pandas as pd
import math


def load_data_multivariate(path_and_filename, flag):
    """
    This function reads data of an external file and uses it to initialize the training data calculate its mean and
    standard deviation. it then converts the data to a numpy-type matrix.

    INPUTS:
        path_and_filename: String representing the name and location of the file.
        flag: number representing if the data should be printed or not.

    OUTPUTS
        x_training: numpy-type matrix with the attributes representing each feature
        y_training: numpy-type vector with the data representing the last-mile cost for each input set of features
        mean: numpy-type vector with the mean of each feature
        std: numpy-type vector with the standard deviation of each feature
    """

    data = []
    x_training = []
    y_training = []
    mean = []
    std = []

    try:
        data = pd.read_csv(path_and_filename)
        df = pd.DataFrame(data, columns=['x1', 'x2', 'x3', 'x4', 'y'])
        x_training = df[['x1', 'x2', 'x3', 'x4']]
        mean = (x_training.mean()).to_numpy()
        # std = (x_training.std()).to_numpy()
        std = np.std(x_training)
        y_training = df[['y']].to_numpy()
    except IOError as e:
        print(e)
        exit(1)

    if flag == 1:
        print('-' * 60)
        print('Training data and Y outputs')
        print('-' * 60)
        print('\t  x1\t\t\tx2\t\t   x3\t\t\tx4\t\t\ty')
        print(data.to_numpy())

    return x_training.to_numpy(), y_training, mean, std


def feature_scaling(x_training, mean, std):
    """
    This function implements feature scaling on the attributes of each feature of the training data-set and prints it
    on the command line.

    INPUTS:
        x_training: numpy-type matrix with the attributes representing each feature
        mean: numpy-type vector with the mean of each feature
        std: numpy-type vector with the standard deviation of each feature

    OUTPUTS
        x_scaled: numpy-type vector with the normalized attributes representing each feature
    """

    num_cols = x_training.shape[1]
    x_scaled = np.zeros_like(x_training)

    for i in range(num_cols):
        scale = (x_training[:, i] - mean[i]) / (math.sqrt((std[i] ** 2) + (10 ** -8)))
        x_scaled[:, i] = scale

    print('-' * 60)
    print('Training data scaled')
    print('-' * 60)
    print(x_scaled)

    return x_scaled


def gradient_descent_multivariate(x_training, y_training, w, stopping_criteria, learning_rate):
    """
    This function calls the function compute_gradient_of_cost_function_multivariate and compute_l2_norm_multivariate
    to implement the multivariate linear regression through a while loop.

    INPUTS:
        x_training_data: numpy-type matrix
        y_training: numpy-type vector
        w: numpy-type vector
        stopping_criteria: number representing when to stop the linear regression
        learning_rate: number representing the learning rated used in the linear regression

    OUTPUTS
        w: numpy-type vector
    """

    num_rows, num_cols = x_training.shape

    x0 = []
    for i in range(num_rows):
        x0.append(1)

    x = np.c_[x0, x_training]
    l2_norm = 100
    i = 0

    while l2_norm > stopping_criteria:
        # compute gradient of cost function
        gradient_cost_function = compute_gradient_of_cost_function_multivariate(x, y_training, w)

        # update parameters
        w = w - learning_rate * gradient_cost_function

        # compute the L2 norm
        l2_norm = compute_l2_norm_multivariate(gradient_cost_function)

        i += 1
    # print('number of iterations: ', i)
    return w


def print_parameters(w):
    """
    This function prints the parameter w on the command line.

    INPUTS:
        w: numpy-type vector with the results of the obtained parameters using multivariate linear regression

    OUTPUTS
        none
    """

    print('-' * 60)
    print('W Parameter')
    print('-' * 60)
    print(w)


def eval_hypothesis_function_multivariate(w, x):
    """
    This function gets the hypothesis needed for the multivariate linear regression.

    INPUTS:
        w: numpy-type vector with the results of the obtained parameters using multivariate linear regression
        x_training_data: numpy-type matrix

    OUTPUTS
        hypothesis_function: numpy-type vector
    """

    hypothesis_function = np.matmul(x, w)
    return hypothesis_function


def compute_gradient_of_cost_function_multivariate(x, y, w):
    """
    This function implements the gradient of cost needed for the multivariate linear regression. It also calls the
    eval_hypothesis_function_multivariate function to then calculate the residual with its value.

    INPUTS:
        x_training_data: numpy-type matrix
        y_training: numpy-type vector
        w: numpy-type vector

    OUTPUTS
        grad_cost_function: numpy-type vector
    """

    grad_cost_function = np.zeros_like(w)
    n = x.shape[0]

    hypothesis_function = eval_hypothesis_function_multivariate(w, x)
    residual = hypothesis_function - y

    for i in range(len(grad_cost_function)):
        grad_cost_function[i] = np.multiply(residual.T[0], (x.T[i])).sum() / n

    return grad_cost_function


def compute_l2_norm_multivariate(gradient_of_cost_function):
    """
    This function gets the l2 norm needed for the multivariate linear regression.

    INPUTS:
        grad_cost_function: numpy-type vector

    OUTPUTS
        l2_norm: Number representing the l2 / euclidean norm
    """

    l2_norm = np.sqrt(np.matmul(gradient_of_cost_function.T, gradient_of_cost_function).sum())
    return l2_norm


def load_testing_data(flag):
    """
    This function reads data of an external file and uses it to initialize the training data calculate its mean and
    standard deviation. it then converts the data to a numpy-type matrix.

    INPUTS:
        flag: number representing if the data should be printed or not.

    OUTPUTS
        x_testing: numpy-type matrix with the attributes representing each feature
        mean: numpy-type vector with the mean of each feature
        std: numpy-type vector with the standard deviation of each feature
    """

    x_test = np.array([[24.51, 0.340, 50.2, 2.78],
                       [33.98, 0.620, 50.1, 9.79],
                       [9.57, 0.320, 52.4, 3.15]])
    n_cols = x_test.shape[1]
    mean = []
    std = []

    for i in range(n_cols):
        mean.append(x_test[:, i].mean())
        std.append(x_test[:, i].std())
        std.append(np.std(x_test[:, i]))

    if flag == 1:
        print('-' * 60)
        print('Testing data')
        print('-' * 60)
        print(x_test)

    return x_test, mean, std


def scale_test(x, mean, std):
    """
    This function implements feature scaling on the attributes of each feature of the testing data-set and prints it
    on the command line.

    INPUTS:
        x: numpy-type matrix with the attributes representing each feature
        mean: numpy-type vector with the mean of each feature
        std: numpy-type vector with the standard deviation of each feature

    OUTPUTS
        x_scaled: numpy-type vector with the normalized attributes representing each feature
    """
    num_cols = x.shape[1]
    x_scaled = np.zeros_like(x)

    for i in range(num_cols):
        test_scale = (x[:, i] - mean[i]) / math.sqrt((std[i] ** 2) + (10 ** -8))
        x_scaled[:, i] = test_scale

    print('-' * 60)
    print('Testing data scaled')
    print('-' * 60)
    print(x_scaled)

    return x_scaled


def predict(w, x_test_scaled):
    """
    This function gets the 'y' (last mile cost) of the testing data.

    INPUTS:
        w: numpy-type vector with the results of the obtained parameters using multivariate linear regression
        x_test_scaled: numpy-type matrix
    OUTPUTS
        y: numpy-type vector
    """

    num_rows = x_test_scaled.shape[0]
    x0 = []

    for i in range(num_rows):
        x0.append(1)

    x_test = np.c_[x0, x_test_scaled]
    y = np.matmul(x_test, w.T[0])

    return y


def print_predictions(price, title):
    """
    This function prints the price predicted on the command line.

    INPUTS:
        price: numpy-type vector representing the price prediced of the last-mile cost
        title: String representing the header of the section printed on the command line

    OUTPUTS
        none
    """

    print('-' * 60)
    print(title)
    print('-' * 60)
    print(price)
