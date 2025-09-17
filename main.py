import time

import numpy as np
import pandas as pd
import tkinter as tk
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageOps

class NeuralNetwork:

    def __init__(self):
        self.data = pd.read_csv('./src/train.csv')
        self.data = np.array(self.data)

        self.m, self.n = self.data.shape
        np.random.shuffle(self.data)
        self.data_dev = self.data[0:1000].T

        self.Y_dev = self.data_dev[0]
        self.X_dev = self.data_dev[1:self.n]
        self.X_dev = self.X_dev / 255.

        self.data_train = self.data[1000:self.m].T
        self.Y_train = self.data_train[0]
        self.X_train = self.data_train[1:self.n]
        self.X_train = self.X_train / 255.
        self._,self.m_train = self.X_train.shape
        self.W1 = np.random.rand(100, 784) - 0.5
        self.b1 = np.random.rand(100, 1) - 0.5
        self.W2 = np.random.rand(10, 100) - 0.5
        self.b2 = np.random.rand(10, 1) - 0.5

        self.one_hot_Y = None
        self.Z = None

        self.Z1 = None
        self.Z2 = None
        self.A1 = None
        self.A2 = None

        self.db1 = None
        self.dW1 = None
        self.dZ1 = None
        self.db2 = None
        self.dW2 = None
        self.dZ2 = None

        self.predictions = None

        self.alpha = 1
        self.iterations = 200


    def ReLU(self):
        self.A1 = np.maximum(self.Z1, 0)


    def softmax(self):
        self.A2 = np.exp(self.Z2) / sum(np.exp(self.Z2))


    def forward(self, X):
        Z1 = self.W1.dot(X) + self.b1
        A1 = np.maximum(Z1, 0)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = np.exp(Z2) / np.sum(np.exp(Z2))
        return np.argmax(A2, 0), A2


    def forward_prop(self):
        self.Z1 = self.W1.dot(self.X_train) + self.b1
        self.ReLU()
        self.Z2 = self.W2.dot(self.A1) + self.b2
        self.softmax()


    def ReLU_deriv(self):
        return self.Z1 > 0


    def one_hot(self):
        self.one_hot_Y = np.zeros((self.Y_train.max() + 1, self.Y_train.size))
        self.one_hot_Y[self.Y_train, np.arange(self.Y_train.size)] = 1


    def backward_prop(self):
        self.one_hot()
        self.dZ2 = self.A2 - self.one_hot_Y
        self.dW2 = 1 / self.m * self.dZ2.dot(self.A1.T)
        self.db2 = 1 / self.m * np.sum(self.dZ2)
        self.dZ1 = self.W2.T.dot(self.dZ2) * self.ReLU_deriv()
        self.dW1 = 1 / self.m * self.dZ1.dot(self.X_train.T)
        self.db1 = 1 / self.m * np.sum(self.dZ1)


    def update_params(self):
        self.W1 = self.W1 - self.alpha * self.dW1
        self.b1 = self.b1 - self.alpha * self.db1
        self.W2 = self.W2 - self.alpha * self.dW2
        self.b2 = self.b2 - self.alpha * self.db2


    def get_predictions(self):
        self.predictions = np.argmax(self.A2, 0)


    def get_accuracy(self):
        return np.sum(self.predictions == self.Y_train) / self.Y_train.size


    def gradient_descent(self):
        for i in range(self.iterations):
            self.forward_prop()
            self.backward_prop()
            self.update_params()
            if i % 10 == 0:
                print("Iteration: ", i)
                self.get_predictions()
                print(f"Accuracy {self.get_accuracy()*100}%")


    def make_predictions(self):
        self.forward_prop()
        self.get_predictions()


    def test_prediction(self, index: int):
        current_image = self.X_train[:, index, None]
        label = self.Y_train[index]

        prediction, _ = self.forward(current_image)
        prediction = prediction[0]
        print("Prediction: ", prediction)
        print("Label:      ", label)

        return current_image, prediction, label
if __name__ == "__main__":
    print("Training start: \n")
    neural_network = NeuralNetwork()
    neural_network.gradient_descent()
