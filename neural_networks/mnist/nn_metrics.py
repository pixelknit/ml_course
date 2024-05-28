import numpy as np

class Metrics:
    def __init__(self):
        pass

    def get_accuracy(self, predictions, y):
        print(predictions, y)
        return np.sum(predictions == y) / y.size

