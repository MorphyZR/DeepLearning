import numpy as np
import tensorflow as tf
import sys, os
sys.path.insert(0, os.getcwd())
from src.base.SimpleCNN import SimpleCNN
from src.common.layer import Affine, Relu, Convolution, Pooling, SoftmaxWithLoss
from src.common.functions import change_to_one_hot
from src.common.optimizer import Adam

import time

(train_img, train_labels), (test_img, test_labels) = tf.keras.datasets.mnist.load_data()
# normalization
train_img = train_img / 255.0
test_img = test_img / 255.0
# reshape
train_img = train_img.reshape(-1, 1, 28, 28)
test_img = test_img.reshape(-1, 1, 28, 28)
# one-hot vector
train_labels = change_to_one_hot(train_labels, 10)
test_labels = change_to_one_hot(test_labels, 10)

iter_num = 100
train_size = train_img.shape[0]
batch_size = 100
lr = 0.001
stride = 1
pad = 1
epoch = train_size // batch_size

# initial network
network = SimpleCNN(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
optimizer = Adam(lr)
loss_list = []
train_acc_list = []
test_acc_list = []
for i in range(iter_num):
    batch_mask = np.random.choice(np.arange(train_size), batch_size)
    batch_img = train_img[batch_mask]
    batch_labels = train_labels[batch_mask]
    
    grads = network.gradient(batch_img, batch_labels)
    optimizer.update(network.params, grads)
    loss = network.loss(batch_img, batch_labels)
    print(f'loss{loss}')
    if i%epoch == 0:
        train_acc = network.accuracy(train_img[:1000], train_labels[:1000])
        test_acc = network.accuracy(test_img[:1000], test_labels[:1000])
        
        print(f'iter:{i} \t loss:{round(loss, 4)} \t train_acc:{round(train_acc, 4)} \t test_acc:{round(test_acc, 4)}')