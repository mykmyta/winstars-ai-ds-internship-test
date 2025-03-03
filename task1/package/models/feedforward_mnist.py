import numpy as np
from package.interfaces.mnist_classifier_interface import MnistClassifierInterface
from tensorflow.keras.layers import Dense, Dropout, Flatten # type: ignore
from tensorflow.keras.models import Sequential # type: ignore

class FeedForwardMnistClassifier (MnistClassifierInterface):
    
    """
    A feedforward neural network classifier for the MNIST dataset.
    Attributes:
        x_train (numpy.ndarray): Training data features.
        y_train (numpy.ndarray): Training data labels.
        x_test (numpy.ndarray): Test data features.
        y_test (numpy.ndarray): Test data labels.
        model (tensorflow.keras.Sequential): The feedforward neural network model.
    Methods:
        __init__(x_train, y_train, x_test, y_test):
            Initializes the classifier with training and test data, and builds the model.
        train():
            Trains the model on the training data for a fixed number of epochs.
        predict(x):
            Predicts the class labels for the given input data.
    """
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(28, 28)))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(10, activation='softmax'))
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    def train(self):
        self.model.fit(self.x_train, self.y_train, epochs=10,verbose=0)
        print("Feed Forward Model trained")
        

    def predict(self, x):
        prediction = self.model.predict(x, verbose=0)
        return np.argmax(prediction, axis=1)