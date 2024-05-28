import numpy as np

class Activations:
    def __init__(self):
        pass

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def softmax(self,x):
        a = np.exp(x) / sum(np.exp(x))
        return a

    def ReLU(self,x):
        return np.maximum(x,0)
