from abc import ABC, abstractmethod


class MnistClassifierInterface(ABC):
    """Common interface for MNIST classifiers used in this project."""

    @abstractmethod
    def train(self) -> None:
        """Train the classifier on the data passed at construction time."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, x):
        """Predict class indices for input samples `x`."""
        raise NotImplementedError
