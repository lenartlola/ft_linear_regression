from linear_regression import LinearRegression
import argparse
from utils import read_data, normalize_data, denormalize_theta, write_thetas

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    verbose = args.verbose
    return verbose


def main():
    verbose = parse_arg()
    km, price = read_data()

    normalized_km, mean_km, std_km = normalize_data(km)
    normalized_price, mean_price, std_price = normalize_data(price)

    model = LinearRegression(learning_rate=0.01, n_iterations=10000)
    model.fit(normalized_km, normalized_price, verbose)

    # Denormalize the found thetas
    denorm_theta0, denorm_theta1 = denormalize_theta(model.theta0, model.theta1, mean_km, std_km, mean_price, std_price)

    # Write the found thetas to a file
    write_thetas(denorm_theta0, denorm_theta1)


if __name__ == '__main__':
    main()
