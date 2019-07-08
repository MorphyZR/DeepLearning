import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

def sigmoid(x: np.array):
    return 1 / (1 + np.exp(-x))

def relu(x: np.array):
    return np.maximum(0, x)

x = np.random.randn(1000, 100)
node_num = x.shape[1]
hidden_layer_size = 5
activations = {}

for i in range(hidden_layer_size):
    if i != 0 :
        x = activations[i-1]
    
    w = np.random.randn(node_num, node_num) * 0.01 # * sqrt(1.0/node_num)
    # w_x = np.random.randn(node_num, node_num) / sqrt(node_num) 
    # w_h = np.random.rand(node_num, node_num) / sqrt(node_num)
    
    z = np.dot(x, w)
    a = sigmoid(z)
    # a = relu((z))
    activations[i] = a

for i,a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(f"{i+1}-layer")
    plt.hist(a.flatten(), 30, range=(0,1))
plt.show()


