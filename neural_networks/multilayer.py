import numpy as np

class Model:
    def __init__(self, aw1=None, aw2=None, ab1=None, zw1=None, zw2=None, zb1=None, nw1=None, nw2=None, nb1=None):
        self.aw1 = aw1
        self.aw2 = aw2
        self.ab1 = ab1
        self.zw1 = zw1
        self.zw2 = zw2
        self.zb1 = zb1
        self.nw1 = nw1
        self.nw2 = nw2
        self.nb1 = nb1

    def init_params(self):

        self.aw1 = np.random.uniform(0,1,1)
        self.aw2 = np.random.uniform(0,1,1)
        self.ab1 = np.random.uniform(0,1,1)
        self.zw1 = np.random.uniform(0,1,1)
        self.zw2 = np.random.uniform(0,1,1)
        self.zb1 = np.random.uniform(0,1,1)
        self.nw1 = np.random.uniform(0,1,1)
        self.nw2 = np.random.uniform(0,1,1)
        self.nb1 = np.random.uniform(0,1,1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward(m, x1, x2):
    a = sigmoid(m.aw1 * x1 + m.aw2 * x2 + m.ab1)
    b = sigmoid(m.zw1 * x1 + m.zw2 * x2 + m.zb1)

    return sigmoid(a * m.nw1 + b * m.nw2 + m.nb1)

def loss_function(m, train_data):
    result = 0
    for i in range(len(train_data)):
        x1 = train_data[i][0]
        x2 = train_data[i][1]
        y = forward(m,x1,x2)
        d = y - train_data[i][2]
        result += d * d

    result /= len(train_data)

    return result

def get_model():
    model = Model()
    model.init_params()
    return model

def difference(m, train_data, eps=0.01):
    g = get_model()
    c = loss_function(m, train_data)

    parms = ['aw1','aw2','ab1','zw1','zw2','zb1','nw1','nw2','nb1']

    for parm in parms:
        orgival_val = getattr(m, parm)
        setattr(m, parm, orgival_val + eps)
        g_val = (loss_function(m, train_data) - c) / eps
        setattr(g, parm, g_val)
        setattr(m, parm, orgival_val)

    return g

def train(m, g, lr):
    parms = ['aw1','aw2','ab1','zw1','zw2','zb1','nw1','nw2','nb1']

    for parm in parms:
        new_val = getattr(m, parm) - lr * getattr(g, parm)
        setattr(m, parm, new_val)

    return m

train_data = [[0,0,0],
              [1,0,1],
              [0,1,1],
              [1,1,0]]

model = get_model()

eps = 1e-1
lr = 1e-1
epochs = 100000

print("Initial loss: ", loss_function(model, train_data))
for epoch in range(epochs):
    g = difference(model, train_data, eps=eps)
    model = train(model, g, lr)
print("Final loss: ", loss_function(model, train_data))

print("--------------------")

for i in range(2):
    for j in range(2):
        print(f"{i}, {j}, {forward(model, i,j)}")


