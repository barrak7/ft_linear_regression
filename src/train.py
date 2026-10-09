"""Training program. Loads model, trains it, and stores parameters."""

import numpy as np

from model import Model
from utils import load_data, write_to_file


class Train:
    """
    The training loop class.

    Loads the data, scales it,
    and sets model parameters.

    A training loop is performed for n Epochs
    or until the loss is small enough.
    """

    def __init__(self) -> None:
        """
        Initialize variables needed to train model.

        Since the Mileage provided in the data file is quite large,
        it is scaled down by a factor of * 1e-5.
        It is expected that is also done during inference.
        """
        self.model: Model = Model()
        self.data: np.ndarray = load_data("data.csv")
        self.x: np.ndarray = self.data[:, 0] * 1e-5
        self.y: np.ndarray = self.data[:, 1]
        self.lr: float = 0.01
        self.n: int = self.y.size
        self.epochs: int = 10000
        self.target_loss = 0.01

    def mse(self, y_hat: np.ndarray) -> float:
        """
        Calculate the mean squared error loss.

        Args:
            y_hat: np.ndarray
                Model predictions

        Returns:
            l: float
                sum((y - y_hat)²) / n
        """
        return np.sum((self.y - y_hat) ** 2) / (self.n * 2)

    def update_weights(self, y_hat: np.ndarray) -> None:
        """Update the weights with one gradient descent step."""
        self.model.w -= self.lr * np.sum((y_hat - self.y) * self.x) / self.n
        self.model.b -= self.lr * np.sum(y_hat - self.y) / self.n

    def training_loop(self) -> None:
        """
        Training loop for n Epochs.

        The trained parameters are saved
        to the `weights` file.
        """
        y_hat: np.ndarray
        loss: float

        for _ in range(self.epochs):
            y_hat = self.model(self.x)

            loss = self.mse(y_hat)

            if loss < self.target_loss:
                break

            self.update_weights(y_hat)
        write_to_file(f"{self.model.w} {self.model.b}", "weights")


def main() -> None:
    """Trigger training loop."""
    t = Train()
    t.training_loop()
    print("Training Complete!")


if __name__ == "__main__":
    main()
