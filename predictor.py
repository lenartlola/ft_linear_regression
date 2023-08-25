from linear_regression import LinearRegression
import os
import sys
from utils import read_thetas


def main():
    t0, t1 = read_thetas("thetas.txt")

    print("Give me a mileage to predict.")
    try:
        mil = float(input("> "))
    except:
        print("Error: maybe a wrong format.")
        sys.exit(127)

    model = LinearRegression()
    predicted = model.predict(mil, t0, t1)

    print("The predicted price: ", round(predicted, 2))


if __name__ == "__main__":
    main()
