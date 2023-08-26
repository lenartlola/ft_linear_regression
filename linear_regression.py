import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio


class LinearRegression:
    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta0 = 0.0
        self.theta1 = 0.0
        self.loss = []

    def _gradient_descent(self, epochs, X, y):
        predicted_y, coe0, coe1, loss = 0.0, 0.0, 0.0, 0.0

        for i in range(epochs):
            predicted_y = self.predict(X[i], self.theta0, self.theta1)
            coe0 += predicted_y - y[i]
            coe1 += (predicted_y - y[i]) * X[i]
            loss += (predicted_y - y[i]) ** 2

        coe0 = self.learning_rate * (1 / epochs) * coe0
        coe1 = self.learning_rate * (1 / epochs) * coe1

        return predicted_y, coe0, coe1, loss

    def fit(self, X, y, verbose=False):
        n = len(X)
        fig, ax = plt.subplots()
        ax.scatter(X, y, color="blue", marker='o', label="Data")
        line, = ax.plot([],[], color="red", label="Regression line")
        ax.legend()
        x_range = np.linspace(min(X), max(X), num=100)
        frames = []

        for _ in range(self.n_iterations):
            pred_y, coe0, coe1, loss = self._gradient_descent(n, X, y)

            self.theta0 -= coe0
            self.theta1 -= coe1

            loss /= (2 * n)
            self.loss.append(loss)

            if verbose and _ % 100 == 0:
                print(f"Iteration {_}, Loss: {loss}, theta0: \
                        {self.theta0}, theta1: {self.theta1}")
                y_range = self.predict(x_range, self.theta0, self.theta1)
                line.set_data(x_range, y_range)
                ax.set_title(f"Iteration {_}, Loss: {loss:.2f}")
                plt.pause(0.1)
                fig.canvas.draw()
                frame_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                frame_data = frame_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frames.append(frame_data)

        if verbose:
            imageio.mimsave("animation.gif", frames, duration=0.7)
            print(f"Final theta0: {self.theta0}, Final theta1: {self.theta1}")

    def predict(self, mileage, theta0, theta1):
        """
        @ predict a price
        :param mileage: km
        :param theta0: trained theta0
        :param theta1: trained theta1
        :return: the predicted price
        """
        # y = b0 + b1 * x
        return theta0 + (theta1 * mileage)
