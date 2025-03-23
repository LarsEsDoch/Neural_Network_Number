import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('./src/train.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape


def init_params():
    W1 = np.random.rand(100, 784) - 0.5
    b1 = np.random.rand(100, 1) - 0.5
    W2 = np.random.rand(10, 100) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2


def ReLU(Z):
    return np.maximum(Z, 0)


def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

print("Training start: \n")
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 1, 500)


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2, show):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction[0])
    print("Label:      ", label)

    if show == 0: return prediction, label
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
    return prediction, label

#test_prediction(3, W1, b1, W2, b2, 1)

'''
while True:
    user_input = input("Gib eine Zahl für den Index ein (oder 'exit' zum Beenden): ")

    if user_input.lower() == "exit":
        print("Programm beendet.")
        break  # Schleife beenden

    if not user_input.isdigit():
        print("Bitte eine gültige Zahl eingeben!")
        continue  # Schleife neu starten

    index = int(user_input)

    if index < 0 or index >= X_train.shape[1]:
        print(f"Bitte eine Zahl zwischen 0 und {X_train.shape[1] - 1} eingeben!")
        continue

    test_prediction(index, W1, b1, W2, b2, 1)
'''

i = 0
total = input("Prediction count: ")
while not total.isdigit():
    total = input("Prediction count: ")
total = int(total)
right = 0
wrong = 0
wrongs = []
print("Predictions \n")
while i < total:
    prediction, label = test_prediction(i, W1, b1, W2, b2, 0)
    if prediction == label:
        print("Right predicted")
        right += 1
    else:
        print("Wrong predicted")
        wrong += 1
        wrongs.append(i)
    i += 1
    print("Accuracy:", (right/i)*100, "%\n")

print("Wrong/Right/Total", wrong, "/", right, "/", total)
s = input("Do you want to see the wrong predictions: ")
if s == "1" or "Yes" or "Ja":
    print("Wrong predictions")
    for wrong_prediction in wrongs :
        index = wrong_prediction
        current_image = X_train[:, index, None]
        prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
        print("Prediction: ", prediction[0])

        current_image = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.show()

while True:
    user_input = input("Gib eine Zahl für den Index ein (oder 'exit' zum Beenden): ")

    if user_input.lower() == "exit":
        print("Programm beendet.")
        break  # Schleife beenden

    if not user_input.isdigit():
        print("Bitte eine gültige Zahl eingeben!")
        continue  # Schleife neu starten

    index = int(user_input)

    if index < 0 or index >= X_train.shape[1]:
        print(f"Bitte eine Zahl zwischen 0 und {X_train.shape[1] - 1} eingeben!")
        continue

    test_prediction(index, W1, b1, W2, b2, 1)