import numpy as np

#neuron
#input x1, x2, x3
#weights w1, w2, w3
#bias b
#learning rate

def loss_function(w,b, train_data):
    result = 0
    count = len(train_data)

    for i in range(count):
        x = train_data[i][0]
        y = x*w + b
        d = y - train_data[i][1]
        result += d * d

    result /= count
    return result

def train(train_data):

    np.random.seed(0)
    w = np.random.uniform(0,10,1)
    b = np.random.uniform(0,5,1)

    epsilon = 0.001
    learning_rate = 0.01
    print("loss: ", loss_function(w,b, train_data))

    epochs = 200
    for i in range(epochs):
        c = loss_function(w, b,train_data)
        cost_distance = (loss_function(w+epsilon,b, train_data) - c) / epsilon
        bias_distance = (loss_function(w,b+ epsilon, train_data) - c) / epsilon

        w -= learning_rate*cost_distance
        b -= learning_rate*bias_distance
        if i % 10 == 0:
            print(loss_function(w,b,train_data))

    print("------------")
    print(w)
    print(b)


if __name__ == "__main__":

    train_data = [[0,0],
                  [1,2],
                  [2,4],
                  [4,8],
                  [8,16]]


    train_example = train(train_data)
