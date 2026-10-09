"""A set of helpful utilities to train and use the model."""

import numpy as np
import pandas as pd


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
