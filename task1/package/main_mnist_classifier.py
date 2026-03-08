from package.models.feedforward_mnist import FeedForwardMnistClassifier
from package.models.cnn_mnist import ConvolutionalMnistClassifier
from package.models.random_forest_mnist import RandomForestMnistClassifier

NN_MODEL = "nn"
CNN_MODEL = "cnn"
RF_MODEL = "rf"


class MnistClassifier:
    """Factory-style wrapper that exposes a unified MNIST classifier API."""

    def __init__(self, model_type, x_train, y_train, x_test, y_test):
        """Initialize a concrete classifier based on the requested model type.

        Args:
            model_type: One of `NN_MODEL`, `CNN_MODEL`, or `RF_MODEL`.
            x_train: Training images.
            y_train: Training labels.
            x_test: Test images.
            y_test: Test labels.

        Raises:
            ValueError: If `model_type` is not supported.
        """
        self.model_type = model_type
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        if self.model_type == NN_MODEL:
            self.model = FeedForwardMnistClassifier(self.x_train, self.y_train, self.x_test, self.y_test)
        elif self.model_type == CNN_MODEL:
            self.model = ConvolutionalMnistClassifier(self.x_train, self.y_train, self.x_test, self.y_test)
        elif self.model_type == RF_MODEL:
            self.model = RandomForestMnistClassifier(self.x_train, self.y_train, self.x_test, self.y_test)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

    def train(self):
        """Train the selected underlying model."""
        self.model.train()

    def predict(self, x):
        """Predict class indices for input samples `x`."""
        return self.model.predict(x)
