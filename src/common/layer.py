import numpy as np
from .functions import *

# Basic Layer =================================
class MulLayer:
    '''In_X * In_Y => Out'''
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y # np.dot(x, y)

    def backward(self, dout):
        '''
            para:
                - dout: the derivative of forward output.
            return:
                - dx : the partial derivative for x
                - dy : the partial derivative for y
        '''
        dx = dout * self.y # np.dot(dout, self.y)
        dy = dout * self.x # np.dot(self.x, dout)

        return dx, dy

class AddLayer:
    ''' In_X + In_Y => Out'''
    def __init__(self):
        pass
    
    def forward(self, x, y):
        return x + y
    
    def backward(self, dout):
        '''
            The backward propagation will directly transfer the derivative to from output to input
            para:
                - dout: the derivative of forward output.
            return:
                - dx : the partial derivative for x
                - dy : the partial derivative for y
        '''
        dx = dout
        dy = dout

        return dx, dy

class Affine:
    ''' y = np.dot(x, w) + b'''
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.x = None
        self.dw = None
        self.db = None

    def forward(self, x: np.array) -> np.array:
        self.x = x
        out = np.dot(x, self.w) + self.b

        return out

    def backward(self, dout: np.array) -> np.array:
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx

# Activation ===============================================
class Relu:
    '''
        Out = max(In_X, 0)
    '''
    def __init__(self):
        self.mask: np.array = None

    def forward(self, x: np.array ) -> np.array:
        # without considering store mask, the followed lines equal to `out = np.where(x <=0, 0, x)`
        self.mask: np.array = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout: np.array) -> np.array:
        dout[self.mask] = 0
        dx = dout 

        return dx

class Sigmoid:
    '''
        Out = 1 / (1 + exp(-x))
    '''
    def __init__(self):
        self.out = None
        self.x = None
        self.w = None
    
    def forward(self, x: np.array) -> np.array:
        out = 1 / (1 + np.exp(-x))
        # the output value will be used in the backward process
        self.out = out
        
        return out

    def backward(self, dout: np.array) -> np.array:
        dx = dout * self.out * (1.0 - self.out)

        return dx

# Output ==============================================

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    
    def forward(self, x, t):
        '''
        x is the training data;
        t is the correct training label;

        return: the loss of current model
        '''
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx

class BatchNorm:
    ''' Batch Normalization'''
    def __init__(self, garma, beta):
        # if the inputs shape = (N, D)
        self.gamma = garma # (D,1)
        self.beta = beta # (D,)
        self.mean = None
        self.deviation = None
        self.square_dev = None
        self.var = None
        self.std = None
        self.std_ = None
        self.x_cap = None

        self.dbeta = None
        self.dgamma = None

    def forward(self, x):
        '''
            x.shape = (N, D)
        '''
        batch_size = float(x.shape[0])
        # step-1: 
        self.mean = np.sum(x, axis=0) / batch_size # (1, D)

        # step-2:
        self.deviation = x - self.mean # (N, D)

        # step-3:
        self.square_dev = self.deviation ** 2 # (N, D)

        # step-4:
        self.var = np.sum(self.square_dev, axis=0) / batch_size # (1, D)

        # step-5
        self.std = np.sqrt(self.var + 1e-7) # (1, D)

        # step-6
        self.std_ = 1. / self.std # (1, D)

        # step-7
        self.x_cap = self.deviation * self.std_  # (N, D)

        # step-8
        y = self.x_cap * self.gamma + self.beta # (N, D)

        return y

    def backward(self, dout: np.ndarray):
        '''
            dout.shape = (N, D)
        '''
        # breakpoint()
        N, D = dout.shape
        # step-8
        self.dbeta = np.sum(dout, axis=0) # (D,)
        dx_cap = dout * self.gamma # elem-wise multiply, (N, D) 
        self.dgamma = np.sum(dout * self.x_cap, axis=0) #(D, )
        
        #step-7
        ddev1 = dx_cap * self.std_ # (N, D)
        dstd_ = np.sum(dx_cap * self.deviation, axis=0) # (1, D)

        # step-6
        dstd = - dstd_ * (self.std_ ** 2) # (1, D)

        # step-5
        dvar = dstd / (2 * self.std) #(1, D)

        # step-4
        dsquare_dev = dvar * np.ones_like(dout)/ float(N) # (N, D)

        # step-3
        ddev2 = dsquare_dev * 2 * self.deviation

        # step-2
        dx1 = ddev1 + ddev2
        dmu = -np.sum(ddev1 + ddev2, axis=0)

        # step-1
        dx2 = dmu * np.ones_like(dout) / float(N)

        # step-0
        dx = dx1 + dx2
        # breakpoint()
        return dx

if __name__ == "__main__":
    pass