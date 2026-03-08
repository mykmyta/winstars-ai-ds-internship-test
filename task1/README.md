# MNIST Classifier Project

This project provides a set of classifiers for the MNIST dataset, including a feedforward neural network, a convolutional neural network (CNN), and a random forest classifier.

## Project Structure

```
task1/
│
├── package/
│   ├── interfaces/
│   │   └── mnist_classifier_interface.py
│   ├── models/
│   │   ├── cnn_mnist.py
│   │   ├── feedforward_mnist.py
│   │   └── random_forest_mnist.py
│   └── main_mnist_classifier.py
├── demo_mnist.ipynb
├── requirements.txt
└── README.md
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd winstars-ai-ds-internship-test/task1
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Models

1. **Run the Jupyter notebook to train the models:**
   ```bash
   jupyter notebook demo_mnist.ipynb
   ```

2. **Follow the instructions in the notebook to train the models and make predictions.**

## Code Overview

### `main_mnist_classifier.py`

This module defines the `MnistClassifier` class, which serves as a wrapper for different types of classifiers (feedforward neural network, CNN, and random forest).

### `models/`

This directory contains the implementation of different classifiers:
- `cnn_mnist.py`: Convolutional Neural Network (CNN) classifier.
- `feedforward_mnist.py`: Feedforward neural network classifier.
- `random_forest_mnist.py`: Random forest classifier.

### `interfaces/mnist_classifier_interface.py`

This module defines the `MnistClassifierInterface`, an abstract base class that all classifiers must implement.



