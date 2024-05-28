import numpy as np

class DataUtils:
    def __init__(self, data):
        self.data = data

    def train_test_split(self, data):
        data = np.array(self.data)
        row, s = data.shape
        np.random.shuffle(data)

        test_data = data[0:2000].T
        y_test = test_data[0]
        x_test = test_data[1:s]
        x_test = x_test / 255.

        train_data = data[2000:row].T
        y_train = train_data[0]
        x_train = train_data[1:s]
        x_train = x_train / 255.

        return x_train, x_test, y_train, y_test
