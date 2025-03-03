import numpy as np
from sklearn.ensemble import RandomForestClassifier
from package.interfaces.mnist_classifier_interface import MnistClassifierInterface

class RandomForestMnistClassifier(MnistClassifierInterface):
    """
    A classifier for the MNIST dataset using a Random Forest model.
    Attributes:
        x_train (numpy.ndarray): Training data features.
        y_train (numpy.ndarray): Training data labels.
        x_test (numpy.ndarray): Test data features.
        y_test (numpy.ndarray): Test data labels.
        model (RandomForestClassifier): The Random Forest model.
    Methods:
        __init__(x_train, y_train, x_test, y_test):
            Initializes the classifier with training and test data.
        train():
            Trains the Random Forest model using the training data.
        predict(x):
            Predicts the labels for the given input data.
    """

    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train.reshape(x_train.shape[0], -1)  # Flatten the input
        self.y_train = y_train
        self.x_test = x_test.reshape(x_test.shape[0], -1)  # Flatten the input
        self.y_test = y_test
        self.model = RandomForestClassifier(n_estimators=300, random_state=42)

    def train(self):
        self.model.fit(self.x_train, self.y_train)
        print("Random Forest Model trained")

   
    
    def predict(self, x):
        x = np.array(x)  
        if x.ndim == 1:  
            x = x.reshape(1, -1)  
        else:  
            x = x.reshape(x.shape[0], -1)  
        return self.model.predict(x)