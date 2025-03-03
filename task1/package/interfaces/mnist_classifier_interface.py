from abc import ABC, abstractmethod

class MnistClassifierInterface(ABC):
    """
    An abstract base class that defines the interface for an MNIST classifier.
    Methods
    -------
    train()
        Abstract method to train the MNIST classifier. Must be implemented by subclasses.
    predict()
        Abstract method to make predictions using the trained MNIST classifier. Must be implemented by subclasses.
    """
    @abstractmethod

    def train():
        pass

    def predict():
        pass