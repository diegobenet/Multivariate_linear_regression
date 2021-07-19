""" multivariate_linear_regression.py
    This python script implements the multivariate linear regression with the methods provided by utilityfunctions.py.

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
import time

# import user-defined libraries
import utilityfunctions as uf


def main():
    """
    This function calls all the methods needed to get data and to implement the multivariate linear regression
    function with said data, and to finally get the last mile cost prediction of our testing data-set.
    It also decides if it should print out the result of certain methods with the flag variable.

    INPUTS:
        none

    OUTPUTS
        none
    """

    # load training data
    # if flag = 1, the training data is printed on the command line
    flag = 1
    x_training, y_training, mean, std = uf.load_data_multivariate('training-data-multivariate.csv', flag)

    # get training data normalized
    x_training_scaled = uf.feature_scaling(x_training, mean, std)

    # initialize hyperparameters
    stopping_criteria = 0.01
    learning_rate = 0.5

    # initialize w
    w = np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])

    # run the gradient descendent method for parameter optimisation purposes
    w = uf.gradient_descent_multivariate(x_training_scaled, y_training, w, stopping_criteria, learning_rate)

    # print w on command line
    uf.print_parameters(w)

    # load training data
    # if flag = 1, the training data is printed on the command line
    x_test, mean_test, std_test = uf.load_testing_data(flag)

    # get testing data normalized
    x_test_scaled = uf.scale_test(x_test, mean, std)

    # predict last-mile costs and print them on command line
    # if flag = 1, the training data is printed on the command line
    price = uf.predict(w, x_test_scaled)
    uf.print_predictions(price, 'Last-mile cost [predicted]')
    print('Run time: ', time.process_time())


main()
