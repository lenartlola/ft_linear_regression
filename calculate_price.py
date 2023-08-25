import numpy as np
import pandas as pd
from linear_regression import LinearRegression
from utils import read_data, read_thetas


def calculate_r_squared(predictions, actual):
    total_variance = np.sum((actual - np.mean(actual)) ** 2)
    residual_variance = np.sum((actual - predictions) ** 2)
    r_squared = 1 - (residual_variance / total_variance)
    return r_squared


def main():
    file_path = "thetas.txt"

    mileage, actual_price = read_data();

    model = LinearRegression(learning_rate=0.01, n_iterations=1000)

    theta0, theta1 = read_thetas(file_path)

    predicted_price = model.predict(mileage, theta0, theta1)
    r_squared = calculate_r_squared(predicted_price, actual_price)

    print("Predicted Prices:", predicted_price)
    print("R-squared:", r_squared)


if __name__ == "__main__":
    main()
