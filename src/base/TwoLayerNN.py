import sys
import os
sys.path.insert(0, os.getcwd())
import numpy as np
from collections import OrderedDict
from src.common.layer import Affine, Relu, SoftmaxWithLoss, BatchNorm
from src.common.optimizer import *
class TwoLayerNN:
    '''
    Using layer implement two layer network
    '''
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01,
        use_batchNorm=True
    ):
        '''
            @param:
                input_size: the num of units in the input layer
                hidden_size: the num of units in the hidden layer
                output_size: the num of units in the output layer
        '''
        self.use_batchNorm = use_batchNorm
        self.params = {}
        # self.__xavier_initial(input_size, hidden_size, output_size)
        self.__he_initial(input_size, hidden_size, output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] =  Affine(self.params['W1'], self.params['B1'])
        if use_batchNorm:
            self.params['gamma1'] = np.ones(hidden_size, dtype=float)
            self.params['beta1'] = np.zeros(hidden_size, dtype=float)
            self.layers['BatchNorm1'] = BatchNorm(
                self.params['gamma1'],
                self.params['beta1']
                )
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['B2'])

        self.lastLayer = SoftmaxWithLoss()
    
    def __weight_initial(self, input_size, hidden_size, output_size, weight_init_std):
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['B1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['B2'] = np.zeros(output_size)

    def __xavier_initial(self, input_size, hidden_size, output_size):
        '''
            initial weight using xavier's method:
                current layer's std = sqrt(1 / previous layer's node num)
            when the activation func using Sigmoid or tanh, it is recommend to use this method
        '''
        self.params['W1'] = np.random.randn(input_size, hidden_size) / np.sqrt(input_size) 
        self.params['B1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        self.params['B2'] = np.zeros(output_size) 

    def __he_initial(self, input_size, hidden_size, output_size):
        '''
                current layer's std = sqrt(2 / previous layer's node num)
            When the activation func using Relu, it is recommend to use this method.
            Can eliminate gradient vanishing.
        '''
        self.params['W1'] = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size) 
        self.params['B1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.params['B2'] = np.zeros(output_size) 

    def predict(self, x:np.ndarray):
        ''' x is the train data'''
        # the predict phase don't need:
        # - softmax layer
        # - dropout
        # - batchnorm
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self, x, t):
        '''
            Calculate the loss.
            x is the training data
            t is the correct one-hot training label
        '''
        y = self.predict(x)

        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        acc = np.sum(y == t) / float(x.shape[0])

        return acc


    def gradient(self,x, t):
        #forward
        self.loss(x, t)

        #backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        #record gradients
        grads = {}
        grads['W1'] = self.layers['Affine1'].dw
        grads['B1'] = self.layers['Affine1'].db
        if self.use_batchNorm:
            grads['gamma1'] = self.layers['BatchNorm1'].dgamma
            grads['beta1'] = self.layers['BatchNorm1'].dbeta
        grads['W2'] = self.layers['Affine2'].dw
        grads['B2'] = self.layers['Affine2'].db

        return grads

        

