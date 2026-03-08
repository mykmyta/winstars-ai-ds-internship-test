import numpy as np
from sklearn.ensemble import RandomForestClassifier
from package.interfaces.mnist_classifier_interface import MnistClassifierInterface


class RandomForestMnistClassifier(MnistClassifierInterface):
    """RandomForest-based classifier for MNIST."""

    def __init__(self, x_train, y_train, x_test, y_test):
        """Flatten image tensors and initialize the RandomForest estimator."""
        self.x_train = x_train.reshape(x_train.shape[0], -1)
        self.y_train = y_train
        self.x_test = x_test.reshape(x_test.shape[0], -1)
        self.y_test = y_test
        self.model = RandomForestClassifier(n_estimators=300, random_state=42)

    def train(self):
        """Fit the RandomForest model on flattened training images."""
        self.model.fit(self.x_train, self.y_train)
        print("Random Forest Model trained")

    def predict(self, x):
        """Predict class indices for one or more samples."""
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        else:
            x = x.reshape(x.shape[0], -1)
        return self.model.predict(x)
