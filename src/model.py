"""The model class basic definition."""


class Model:
    """The linear model class definition."""

    def __init__(self, w: float = 0, b: float = 0) -> None:
        """
        Initialize the model's weight and bias.

        Args:
            w: float
                The weight of the model.
            b: float
                The bias of the model.

        Returns:
            None
        """
        self.w: float = w
        self.b: float = b

    def forward(self, x: int) -> float:
        """
        Forward pass of the model.

        Args:
            x: int
                The input data.

        Returns:
            float:
                The output of the model.
        """
        return self.w * x + self.b
