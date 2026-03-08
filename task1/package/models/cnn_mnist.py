import numpy as np
from package.interfaces.mnist_classifier_interface import MnistClassifierInterface
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore


class ConvolutionalMnistClassifier(MnistClassifierInterface):
    """Convolutional neural network classifier for MNIST."""

    def __init__(self, x_train, y_train, x_test, y_test):
        """Initialize data holders and build a compact CNN architecture."""
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation="relu"))
        self.model.add(Conv2D(128, (2, 2), activation="relu"))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(10, activation="softmax"))
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    def train(self):
        """Train the CNN on the MNIST training split."""
        self.model.fit(self.x_train, self.y_train, epochs=10, verbose=0)
        print("CNN Model trained")

    def predict(self, x):
        """Predict class indices for one or more input images."""
        prediction = self.model.predict(x, verbose=0)
        return np.argmax(prediction, axis=1)
