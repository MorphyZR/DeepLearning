'''
implement a two lasyer NN with layers
'''
import sys
import os
sys.path.insert(0, os.getcwd())
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from src.base.TwoLayerNN import TwoLayerNN
from src.common.functions import change_to_one_hot
from src.common.optimizer import *

# load data
print("load data...")
(train_imgs, train_labels), (test_imgs, test_labels) = tf.keras.datasets.mnist.load_data()

# hyperparams
iteration_num = 10000
train_size = train_imgs.shape[0]
test_size = test_imgs.shape[0]
batch_size = 100
epoch = max(train_size // batch_size, 1)
learning_rate = 0.001
network = TwoLayerNN(784, 50, 10, use_batchNorm=False)
# optimizer = SGD(learning_rate)
# optimizer = Momentum(learning_rate)
# optimizer = AdaGrad(learning_rate)
optimizer = Adam(learning_rate) # cannot learning normally when lr = 0.1

# flatten
train_imgs = train_imgs.reshape(train_size, -1)
test_imgs = test_imgs.reshape(test_size, -1)
# one-hot
train_labels = change_to_one_hot(train_labels, 10)
test_labels = change_to_one_hot(test_labels, 10)
# normalize
train_imgs = train_imgs / 255.0
test_imgs = test_imgs / 255.0

# records
loss_list = []
train_acc_list = [network.accuracy(train_imgs, train_labels)]
test_acc_list = [network.accuracy(test_imgs, test_labels)]
for i in range(iteration_num):
    # batch
    batch_mask = np.random.choice(train_size, size=batch_size)
    batch_train = train_imgs[batch_mask]
    batch_labels = train_labels[batch_mask]
    # gradient descent
    grads = network.gradient(batch_train, batch_labels)

    # Optimizer: update params
    optimizer.update(network.params, grads)

    # log loss
    loss = network.loss(batch_train, batch_labels)
    loss_list.append(loss)

    # records accuracy
    if i % epoch == 0:
        train_acc = network.accuracy(train_imgs, train_labels)
        train_acc_list.append(train_acc)

        test_acc = network.accuracy(test_imgs, test_labels)
        test_acc_list.append(test_acc)

        print(f'iter:{i} \tloss:{loss} \ttrain_acc:{train_acc} \ttest_acc:{test_acc}')

plt.plot(loss_list)
# plt.plot(train_acc_list, 'b-')
# plt.plot(test_acc_list, 'r-')
plt.show()
