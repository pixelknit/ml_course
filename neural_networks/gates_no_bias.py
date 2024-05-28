import numpy as np
import pandas as pd

def loss_function(w1,w2,train_data):
    result = 0
    count = len(train_data)

    for i in range(count):
        x1 = train_data[i][0]
        x2 = train_data[i][1]
        y = sigmoid(x1*w1 + x2*w2)
        d = y - train_data[i][2]
        result += d * d

    result /= count
    return result

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train(train_data):

    #np.random.seed(0)
    w1 = np.random.uniform(0,1,1)
    w2 = np.random.uniform(0,1,1)

    loss_values = []

    epochs = 500
    eps = 0.01
    lr = 0.1

    print("initial loss: ", loss_function(w1, w2, train_data))

    for i in range(epochs):
        c = loss_function(w1,w2,train_data)
        #print(f"W1 = {w1}, W2 = {w2}, loss = {c}")
        dw1 = (loss_function(w1 + eps, w2, train_data) - c) / eps
        dw2 = (loss_function(w1, w2 + eps, train_data) - c) / eps
        bd = (loss_function(w1,w1+eps, train_data)-c) / eps

        w1 -= lr * dw1
        w2 -= lr * dw2
        loss_values.append(c)

    print("-------")
    print(f"W1 = {w1}, W2 = {w2}, Loss = {loss_function(w1,w2, train_data)}")
    print("-------")

    #test preds
    for i in range(2):
        for j in range(2):
            print(f"{i} | {j} = {sigmoid(i*w1 + j*w2)}")

    return loss_values

if __name__ == "__main__":

    train_data = [[0,0,0],
                  [1,0,1],
                  [0,1,1],
                  [1,1,1]]

    loss_values = train(train_data)

    df = pd.DataFrame(loss_values)
    df.to_csv("loss_values.csv", index=False)
