from linear_regression import LinearRegression
import os
import sys


def main():
    file_path = "thetas.txt"
    t0 = 0.0
    t1 = 0.0

    if os.path.exists(file_path):
        with open(file_path) as f:
            t0 = float(f.readline())
            t1 = float(f.readline())

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
