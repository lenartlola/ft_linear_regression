import pandas as pd


def estimation(t0: float, t1: float, mil) -> float:
    return t0 + (mil * t1)


# estimateP rice(mileage) = θ0 + (θ1 ∗ mileage)
def predict(t0: float, t1: float):
    user_input = input("Give me a mileage please: ")
    mil = float(user_input)
    # intercept + mileage * slope
    print(f"The predicted value for {user_input} is: '{t0 + (mil * t1)} CHF'")


def train(values, t0, t1, l_rate):
    """Do one iteration of training."""
    delta_intercept = l_rate * sum([estimation(n[0], t0, t1) - n[1] for n in values]) / len(values)
    delta_slope = l_rate * sum([(estimation(n[0], t0, t1) - n[1]) * n[0] for n in values]) / len(values)
    return delta_intercept, delta_slope


#                              m - 1
# tmpθ0 = learningRate ∗ 1 / m   ∑    (estimateP rice(mileage[i]) − price[i])
#                              i = 0
#                              m - 1
# tmpθ1 = learningRate ∗ 1 / m   ∑    (estimateP rice(mileage[i]) − price[i]) ∗ milleage[i]
#                              i = 0


def train_model(data, epochs: int, l_rate: float, t0: float, t1: float) -> tuple[float, float]:
    km = data.iloc[:, 0]
    price = data.iloc[:, 1]

    max_km = max(km)
    max_price = max(price)

    t0 /= max_price
    t1 *= max_km / max_price
    values = list(zip(km / max_km, price / max_price))
    for n in range(epochs):
        t0_d, t1_d = train(values, t0, t1, l_rate)
        t0 -= t0_d
        t1 -= t1_d
    return t0 * max_price, (t1 * max_price) / max_km


def read_data(path):
    """Load csv data from path."""
    ret_data = pd.read_csv(path)
    return ret_data


def main():
    t0: float = 0
    t1: float = 0
    data = read_data("./data.csv")
    epochs: int = 15
    bias: float = 0.1
    # predict(t0, t1)
    t0, t1 = train_model(data, epochs, bias, t0, t1)
    predict(t0, t1)


if __name__ == "__main__":
    main()
