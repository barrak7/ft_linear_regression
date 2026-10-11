"""A set of helpful utilities to train and use the model."""

import numpy as np
import pandas as pd
from matplotlib import axes


def load_data(file: str) -> np.ndarray:
    """
    Load data from csv `file` and return ndarray.

    Args:
        file: str
            The path of the file.

    Returns:
        np.ndarray
    """
    try:
        df: pd.DataFrame = pd.read_csv(file)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Load Data: {e}!")

    return df.to_numpy()


def write_to_file(text: str, file: str) -> None:
    """Write `text` to `file`."""
    try:
        with open(file, "w") as f:
            f.write(text)

    except PermissionError as e:
        raise SystemExit(f"Write to file: {e}!")

    except OSError as e:
        raise SystemExit(f"Write to file: {e}!")


def read_file(file: str) -> str:
    """Read file and return contents."""
    res: str = ""
    try:
        with open(file, "r") as f:
            for line in f:
                res += line
        return res
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Read File: {e}!")


def plot_data(ax: axes, x: np.ndarray, y: np.ndarray) -> None:
    """
    Take x and y data points and plot them on ax.

    Args:
        ax: matplotlib.ax
            sub-plot for our data
        x: np.ndarray, y: np.ndarray
            x, y data points
    """
    ax.scatter(x, y, label="Data")
    ax.set_xlabel("Mileage")
    ax.set_ylabel("Price")
    ax.set_title("Car Price by Mileage")


def plot_line(ax: axes, a: tuple, b: tuple) -> None:
    """Plot line on plot `ax` going through points `a` and `b`."""
    ax.axline(a, b, label="Regression Line")


def plot_loss(ax: axes, losses: list[float], epochs: list[int]) -> None:
    """Plot loss change through epochs."""
    ax.plot(epochs, losses)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("MSE Loss")
    ax.set_title("MSE Loss per epoch.")
