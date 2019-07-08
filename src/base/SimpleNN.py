import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath('.'))
from src.common.functions import *

class SimpleThreeLayerNet:
    '''
        This class will realize a simple, three layers neural network;
        - The input layer has 2 input, each input is a vector with 3 elements;
        - The hidden layer has 3 union;
        - the the output layer has 2 output;
    '''
    def __init__(self):
        # input layer weights     
        # w.shape = (#input_units, #output_units)
        # b.shape = (1, #output_units)
        self.w1 = np.array([
            [0.1, 0.3, 0.5],
            [0.2, 0.4, 0.6],
        ])
        self.b1 = np.array([0.1, 0.2, 0.3])

        # first hidden layer weights
        self.w2 = np.array([
            [0.1, 0.4],
            [0.2, 0.5],
            [0.3, 0.6]
        ])
        self.b2 = np.array([0.1, 0.3])

        # second hidden layer weights
        self.w3 = np.array([
            [0.1, 0.4],
            [0.4, 0.3]
        ])
        self.b3 = np.array([0.2, 0.4])



    def forward(self, x: np.array) -> np.array:
        w1, b1, w2, b2, w3, b3 = self.w1, self.b1, self.w2, self.b2, self.w3, self.b3

        # input layer
        a1 = np.dot(x, w1) + b1
        a1 = sigmoid(a1)

        # first hidden layer
        a2 = np.dot(a1, w2) + b2
        a2 = sigmoid(a2)

        # second hidden layer
        a3 = np.dot(a2, w3) + b3
        a3 = relu(a3)

        # output layer
        out = softmax(a3)
        return out

class SimpleNet:
    '''
        Simple two layer model
    '''
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        # input layer
        # N~(0, weight_init_std**2)
        self.params['w1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)

        # output layer
        self.params['w2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)


    def predict(self, x: np.array) -> np.array:
        # input
        
        a1 = np.dot(x, self.params['w1']) + self.params['b1']
        z1 = sigmoid(a1)

        # output
        a2 = np.dot(z1, self.params['w2']) + self.params['b2']
        y = softmax(a2)

        return y

    def loss(self, x: np.array, t: np.array) -> float:
        '''
            x is the input, t is the correct tags.
        '''
        
        z = self.predict(x)
        y = softmax(z)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        '''
            calculate current accuracy
        '''
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # def numerical_gradient(self, x, t):
    #     '''
    #         calculate the gradient of params
    #     '''
    #     loss_W = lambda W : self.loss(x, t)

    #     grad = {}
    #     grad['w1'] = numerical_gradient(loss_W, self.params['w1'])
    #     grad['b1'] = numerical_gradient(loss_W, self.params['b1'])
    #     grad['w2'] = numerical_gradient(loss_W, self.params['w2'])
    #     grad['b2'] = numerical_gradient(loss_W, self.params['b2'])

    #     return grad

    def gradient(self, x, t):
        W1, W2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        # breakpoint()
        dy = (y - t) / batch_num
        grads['w2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['w1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


if __name__ == "__main__":
    # test SimpleThreeLayerNete
    # input = np.array([1,4])
    # nn = SimpleThreeLayerNet()
    # print(f"simple NN {nn.forward(input)}")

    # ===================================================
    # test GadientSimpleNet
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from src.dataset.mnist import load_mnist
    # load handwritten data
    minst = tf.keras.datasets.mnist
    (train_imgs, train_labels), (test_imgs, test_labels) = minst.load_data()
    # flatten
    train_imgs = train_imgs.reshape(train_imgs.shape[0], -1)
    test_imgs = test_imgs.reshape(test_imgs.shape[0], -1)

    # normalize
    train_imgs = train_imgs.astype(np.float32)
    train_imgs /= 255.0
    test_imgs = test_imgs.astype(np.float32)
    test_imgs /= 255.0

    # change label to one-hot-vector
    train_labels = change_to_one_hot(train_labels, 10)
    test_labels = change_to_one_hot(test_labels, 10)

    # (train_imgs, train_labels), (test_imgs, test_labels) = load_mnist(normalize=True, one_hot_label=True)


    # hyper parameters
    iters_num = 10000
    train_size = train_imgs.shape[0]
    batch_size = 100
    iter_per_epoch = max(train_size / batch_size, 1)
    learning_rate = 0.1
    network = SimpleNet(784, 50, 10)

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    # batch
    for i in range(iters_num):
        # mini-batch
        batch_mask = np.random.choice(train_size, batch_size)
        batch_imgs = train_imgs[batch_mask]
        batch_labels = train_labels[batch_mask]
        # gradient descent
        grad = network.gradient(batch_imgs, batch_labels)

        # update weights
        for key, value in grad.items():
            network.params[key] -= learning_rate * value

        loss = network.loss(batch_imgs, batch_labels)
        train_loss_list.append(loss)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(train_imgs, train_labels)
            test_acc = network.accuracy(test_imgs, test_labels)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(f"iter: {i}, loss: {loss}, train_acc:{train_acc}, test_acc{test_acc}")
            

    # plt.plot(train_loss_list)
    plt.plot(test_acc_list, 'r-')
    plt.plot(train_acc_list, 'b-')
    plt.show()

        

