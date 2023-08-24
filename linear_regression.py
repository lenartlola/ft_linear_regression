import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class LinearRegression:
    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta0 = 0.0
        self.theta1 = 0.0
        self.loss = []

    @staticmethod
    def _mean_square_error(y, y_hat):
        error = np.mean((y - y_hat)**2)
        return error

    def fit(self, X, y, verbose=False):
        n = len(X)

        for _ in range(self.n_iterations):
            tmp_theta0, tmp_theta1 = 0.0, 0.0
            loss = 0.0

            for i in range(n):
                predicted_y = self.predict(X[i], self.theta0, self.theta1)
                tmp_theta0 += predicted_y - y[i]
                tmp_theta1 += (predicted_y - y[i]) * X[i]
                loss += (predicted_y - y[i]) ** 2

            tmp_theta0 = self.learning_rate * (1 / n) * tmp_theta0
            tmp_theta1 = self.learning_rate * (1 / n) * tmp_theta1

            self.theta0 -= tmp_theta0
            self.theta1 -= tmp_theta1

            loss /= (2 * n)  # Dividing by 2m to match the cost function formula
            self.loss.append(loss)

            if verbose and _ % 100 == 0:
                print(f"Iteration {_}, Loss: {loss}, theta0: \
                        {self.theta0}, theta1: {self.theta1}")

        print(f"Final theta0: {self.theta0}, Final theta1: {self.theta1}")

    def predict(self, mileage, theta0, theta1):
      # y = b0 + b1 * x
      return theta0 + (theta1 * mileage)
