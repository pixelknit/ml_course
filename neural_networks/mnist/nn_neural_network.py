import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nn_data_utils import DataUtils
from nn_metrics import Metrics
from nn_encoders import Encoders
from nn_activations import Activations

class NeuralNetwork:
    def __init__(self, data, samples=None, elements=None):
        self.data = data
        self.samples, self.elements = self.data.shape

    def initialize_parameters(self):
        w1 = np.random.rand(10, 784) - 0.5
        b1 = np.random.rand(10, 1) - 0.5
        w2 = np.random.rand(10,10) - 0.5
        b2 = np.random.rand(10,1) - 0.5

        return w1,b1,w2,b2

    def ReLU(self, x):
        a = Activations()
        return a.ReLU(x)

    def softmax(self,x):
        a = Activations()
        return a.softmax(x)

    def forward(self, w1,b1,w2,b2,x):
        x1 = w1.dot(x) + b1
        a1 = self.ReLU(x1)
        x2 = w2.dot(a1) + b2
        a2 = self.softmax(x2)

        return x1, a1, x2, a2

    def one_hot_encoder(self,y):
        a = Encoders()
        return a.one_hot_encoder(y)

    def backward(self, x1, a1, x2, a2, w1, w2, x, y):
        a = y.size
        one_hot_y = self.one_hot_encoder(y)
        dx2 = a2 - one_hot_y
        dw2 = 1 / a * dx2.dot(a1.T)
        db2 = 1 / a * np.sum(dx2)
        relu_derivative = x1 > 0
        dx1 = w2.T.dot(dx2) * relu_derivative
        dw1 = 1 / a * dx1.dot(x.T)
        db1 = 1 / a * np.sum(dx1)

        return dw1, db1, dw2, db2

    def step(self, w1,b1,w2,b2,dw1, db1, dw2, db2, lr=0.01):
        w1 -= lr * dw1
        b1 -= lr * db1
        w2 -= lr * dw2
        b2 -= lr * db2

        return w1, b1, w2, b2

    def get_predictions(self, x):
        return np.argmax(x,0)

    def get_accuracy(self, preds, y):
        a = Metrics()
        return a.get_accuracy(preds, y)

    def gradient_descent(self, x,y,lr=0.01,epochs=5):
        w1, b1, w2, b2 = self.initialize_parameters()
        for epoch in range(epochs):
            x1, a1, x2, a2 = self.forward(w1, b1, w2, b2, x)
            dw1, db1, dw2, db2 = self.backward(x1, a1, x2, a2, w1, w2, x, y)
            w1, b1, w2, b2 = self.step(w1, b1, w2, b2, dw1, db1, dw2, db2, lr)
            if epoch % 10 == 0:
                print("Epoch: ", epoch +1)
                preds = self.get_predictions(a2)
                print(self.get_accuracy(preds, y))

        return w1, b1, w2, b2


if __name__ == "__main__":
    data = pd.read_csv("train.csv")
    neural_model = NeuralNetwork(data)
    dataset = DataUtils(data)

    X_train, X_test, y_train, y_test = dataset.train_test_split(data)

    w1,b1,w2,b2 = neural_model.gradient_descent(X_train, y_train, 0.1, 500)

    def make_predictions(X, W1, b1, W2, b2):
        _, _, _, A2 = neural_model.forward(W1, b1, W2, b2, X)
        predictions = neural_model.get_predictions(A2)
        return predictions

    def test_prediction(index, W1, b1, W2, b2):
        current_image = X_train[:, index, None]
        prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
        label = y_train[index]
        print("Prediction: ", prediction)
        print("Label: ", label)

        current_image = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.show()

    test_prediction(0, w1, b1, w2, b2)



