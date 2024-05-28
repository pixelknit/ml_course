import numpy as np

class Encoders:
    def __init__(self):
        pass

    def one_hot_encoder(self, y):
        one_hot = np.zeros((y.size, y.max()+1))
        one_hot[np.arange(y.size), y] = 1
        one_hot = one_hot.T
        return one_hot
