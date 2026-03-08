from package.models.feedforward_mnist import FeedForwardMnistClassifier
from package.models.cnn_mnist import ConvolutionalMnistClassifier
from package.models.random_forest_mnist import RandomForestMnistClassifier

NN_MODEL = 'nn'
CNN_MODEL = 'cnn'
RF_MODEL = 'rf'

class MnistClassifier:
    """
    A classifier for the MNIST dataset that supports different model types.
    Attributes:
        model_type (str): The type of model to use (NN_MODEL, CNN_MODEL, RF_MODEL).
        x_train (array-like): Training data features.
        y_train (array-like): Training data labels.
        x_test (array-like): Testing data features.
        y_test (array-like): Testing data labels.
        model (object): The instantiated model based on the model_type.
    Methods:
        train():
            Trains the model using the training data.
        predict(x):
            Predicts the labels for the given input data.
    Raises:
        ValueError: If an invalid model type is provided.
    """
    def __init__(self, model_type, x_train, y_train, x_test, y_test):
        """
        Initializes the MnistClassifier with the given model type and data.
        
        Parameters:
            model_type (str): The type of model to use ('NN_MODEL', 'CNN_MODEL', 'RF_MODEL').
            x_train (array-like): Training data features.
            y_train (array-like): Training data labels.
            x_test (array-like): Testing data features.
            y_test (array-like): Testing data labels.
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
        """
        Trains the model using the training data.
        """
        self.model.train()
        
    def predict(self, x):
        return self.model.predict(x)


