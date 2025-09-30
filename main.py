import os
import sys
from tkinter import ttk

import numpy as np
import pandas as pd
import tkinter as tk
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageOps


def resource_path(relative_path: str) -> str:
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class NeuralNetwork:

    def __init__(self):
        self.data = pd.read_csv(resource_path('./src/train.csv'))
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
        self.iterations = 10000

    def ReLU(self):
        self.A1 = np.maximum(self.Z1, 0)

    def softmax(self):
        self.A2 = np.exp(self.Z2) / sum(np.exp(self.Z2))

    def forward(self, X):
        Z1 = self.W1.dot(X) + self.b1
        A1 = np.maximum(Z1, 0)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = np.exp(Z2) / np.sum(np.exp(Z2))
        return np.argmax(A2, 0)[0], A2

    def forward_prop(self, X_train):
        self.Z1 = self.W1.dot(X_train) + self.b1
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

    def gradient_descent(self, iterations: int = None):
        if iterations is None:
            iterations = self.iterations
        print("Training start: \n")
        for i in range(iterations):
            self.forward_prop(self.X_train)
            self.backward_prop()
            self.update_params()
            if i % 10 == 0:
                print("Iteration: ", i+10)
                self.get_predictions()
                print(f"Accuracy {self.get_accuracy()*100}%")

    def make_predictions(self):
        self.forward_prop(self.X_dev)
        self.get_predictions()

    def test_prediction(self, index: int):
        current_image = self.X_dev[:, index, None]
        label = self.Y_dev[index]

        prediction, _ = self.forward(current_image)
        print("Prediction: ", prediction)
        print("Label:      ", label)

        return current_image, prediction, label

    def save_model(self, path: str = "model_weights.npz"):
        path = resource_path(path)
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
        print("Model saved to", path)

    def load_model(self, path: str = "model_weights.npz"):
        path = resource_path(path)
        data = np.load(path)
        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]
        print("Model loaded from", path)


class UI:

    def __init__(self, neuralNetwork: NeuralNetwork, close_automatically: bool = False, wait_time: int = 2):
        self.neural_network = neuralNetwork
        self.close_automatically = close_automatically
        self.wait_time = wait_time

    def show_prediction(self, current_image):
        current_image = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.show(block=False)
        if self.close_automatically:
            plt.pause(self.wait_time)
            plt.close()

    def predict_multiple(self):
        total = input("Prediction count: ")
        while not total.isdigit():
            total = input("Prediction count: ")
        total = int(total)
        right = 0
        wrong = 0
        wrongs = []
        print("Predictions \n")
        for i in range(total):
            _, prediction, label = self.neural_network.test_prediction(i)
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
        s = input("Do you want to see the wrong predictions (Y/n): ")
        if s.lower() == "y":
            print("Wrong predictions")
            for wrong_prediction in wrongs :
                current_image, prediction, label = self.neural_network.test_prediction(wrong_prediction)
                self.show_prediction(current_image)
                print("Prediction: ", prediction)
                print("Label:", label)

    def predict_loop(self):
        while True:
            user_input = input(f"Please enter a number for the index (Max: {self.neural_network.X_dev.shape[1]}) (or 'exit' to end): ")

            if user_input.lower() == "exit":
                print("Programm endet.")
                break

            if not user_input.isdigit():
                print("Please enter a number!")
                continue

            index = int(user_input)

            if index < 0 or index >= self.neural_network.X_dev.shape[1]:
                print(f"Enter a number between 0 and {self.neural_network.X_dev.shape[1] - 1}!")
                continue

            current_image, _, _ = self.neural_network.test_prediction(index)
            self.show_prediction(current_image)


class PaintApp:

    def __init__(self, neuralNetwork: NeuralNetwork):
        self.root = tk.Tk()
        self.neural_network = neuralNetwork

        self.root.title("Detect Number")

        self.canvas_size = 280
        self.image_size = 28

        self.last_x, self.last_y = None, None

        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.pack()

        self.button_clear = tk.Button(self.root, text="Clear", command=self.clear_canvas)
        self.button_clear.pack()

        print("Press c to clear canvas")
        print("Press i to show the input")
        self.root.bind("<c>", lambda event: self.clear_canvas())
        self.root.bind("<i>", lambda event: self.show_input())

        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonPress-1>", self.start_pos)
        self.canvas.bind("<ButtonRelease-1>", self.reset_last_pos)

        self.bar_labels = None
        self.bars = None
        self.bar_frames = None
        self.label_pred = None
        self.conf_window = None

        self.create_confidence_window()

        self.root.mainloop()

    def create_confidence_window(self):
        self.conf_window = tk.Toplevel(self.root)
        self.conf_window.title("Prediction Confidence")

        self.label_pred = tk.Label(self.conf_window, text="Prediction: None", font=("Arial", 24))
        self.label_pred.pack(pady=10)

        self.bar_frames = []
        self.bars = []
        self.bar_labels = []

        for i in range(10):
            frame = tk.Frame(self.conf_window)
            frame.pack(fill="x", padx=10, pady=2)

            tk.Label(frame, text=str(i), width=2).pack(side="left")
            bar = ttk.Progressbar(frame, length=200, maximum=1.0)
            bar.pack(side="left", padx=5)
            label = tk.Label(frame, text="0.00%")
            label.pack(side="left")

            self.bar_frames.append(frame)
            self.bars.append(bar)
            self.bar_labels.append(label)

    def start_pos(self, event):
        self.last_x, self.last_y = event.x, event.y

    def reset_last_pos(self, event):
        self.last_x, self.last_y = None, None

    def paint(self, event):
        x, y = event.x, event.y
        r = 10

        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(self.last_x, self.last_y, x, y, fill="black", width=r * 2, capstyle=tk.ROUND,
                                    smooth=True)

            self.draw.line([self.last_x, self.last_y, x, y], fill=0, width=r * 2)
        else:
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
            self.draw.ellipse([x - r, y - r, x + r, y + r], fill=0)

        self.last_x, self.last_y = x, y

        self.predict_digit()

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image)

        self.label_pred.config(text=f"Prediction: None")

        for i in range(10):
            prob = 0
            self.bars[i]['value'] = prob
            self.bar_labels[i].config(text=f"{prob * 100:.2f}%")

    def preprocess_image(self):
        img = self.image.convert("L")
        img = ImageOps.invert(img)

        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)

        img.thumbnail((20, 20), Image.Resampling.LANCZOS)

        new_img = Image.new("L", (28, 28), 0)
        x = (28 - img.size[0]) // 2
        y = (28 - img.size[1]) // 2
        new_img.paste(img, (x, y))

        img_array = np.array(new_img).reshape(784, 1) / 255.
        return img_array

    def predict_digit(self):
        img_array = self.preprocess_image()
        prediction, probabilities = self.neural_network.forward(img_array)

        self.label_pred.config(text=f"Prediction: {prediction}")

        for i in range(10):
            prob = probabilities[i, 0]
            self.bars[i]['value'] = prob
            self.bar_labels[i].config(text=f"{prob * 100:.2f}%")

    def show_input(self):
        img_array = self.preprocess_image()
        current_image = img_array.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.show(block=False)

if __name__ == "__main__":
    neural_network = NeuralNetwork()
    neural_network.load_model()
    #neural_network.gradient_descent(10000)
    #neural_network.save_model("model.npz")
    #ui = UI(neural_network, True, 5)
    #ui.predict_loop()
    #ui.predict_multiple()
    app = PaintApp(neural_network)
