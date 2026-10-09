"""Runs the trained model on user input."""

from model import Model
from utils import read_file


def estimate(mileage: int) -> int:
    """
    Predcit the price of a car based on mileage.

    The trained model parameters are loaded and used.

    The predicted price is converted from ndarray[float]
    to int.
    """
    params: str = read_file("weights")
    model: Model = Model()
    res: int = 0

    try:
        if params:
            model.w, model.b = map(float, params.split())
        res = int(model(mileage * 1e-5))
        return res
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Estimate: {e}!")


def main() -> None:
    """Prompt the user for input and predict price."""
    mileage: str = input("Please enter the car mileage: ")

    try:
        print("The estimated price is: ", estimate(int(mileage)))
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Estimate Main: {e}!")


if __name__ == "__main__":
    main()
