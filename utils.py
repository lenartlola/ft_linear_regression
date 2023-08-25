from linear_regression import LinearRegression
import sys
import pandas as pd
import numpy as np
import argparse
import os

def normalize_data(data):
    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean) / std
    return normalized_data, mean, std


def denormalize_theta(theta0, theta1, mean_x, std_x, mean_y, std_y):
    denorm_theta0 = mean_y + theta0 * std_y - theta1 * (mean_x * std_y / std_x)
    denorm_theta1 = theta1 * (std_y / std_x)
    return denorm_theta0, denorm_theta1


def read_data():
    try:
        data = pd.read_csv("data.csv")
    except:
        print("Could not open/read file data.csv")
        sys.exit(127)
    X = np.array(data["km"])
    Y = np.array(data["price"])

    return X, Y


def read_thetas(file_path):
    t0 = 0.0
    t1 = 0.0

    if os.path.exists(file_path):
        with open(file_path) as f:
            t0 = float(f.readline())
            t1 = float(f.readline())

    return t0, t1

def write_thetas(t0, t1):
    file_path = "thetas.txt"

    if os.path.exists(file_path):
        os.remove(file_path)
    try:
        with open(file_path, "a") as f:
            thetas = [str(t0), str(t1)]
            f.write('\n'.join(thetas))
            f.write('\n')
            f.close()
    except IOError:
        print("Couldn't create a new file thetas.txt")
        raise