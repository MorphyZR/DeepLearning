import numpy as np
from .functions import *
from .util import im2col, col2im

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
        self.original_x_shape = None

    def forward(self, x: np.array) -> np.array:
        
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        # breakpoint()
        out = np.dot(self.x, self.w) + self.b

        return out

    def backward(self, dout: np.array) -> np.array:
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape) 
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

class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            # random drop some input
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            # if it is not training, the input still need multiple the dropout ratio
            return x * (1 - self.dropout_ratio)

    def backward(self, dout):
        # the dropped neral do not nned backward
        return dout * self.mask

class Convolution:
    def __init__(self, filter, offset, stride=1, pad=0):
        '''
        @params:
        - filter: the filter(kernel) of convolution layer, 4-D array(FN,C,FH,FW)
        - offset: the offset of convolution layer,
        '''
        self.filter = filter
        self.offset = offset
        self.stride = stride
        self.pad = pad

        # needed by backward
        self.x = None
        self.col = None
        self.flatten_filter = None

        # backward result
        self.dW = None
        self.db = None

    def forward(self, x):
        '''
        @params:
        - x : the inpud data, 4-D array, (N : bactch size, C: channel num, H, W)
        
        @return:
        
        '''
        FN, C, FH, FW = self.filter.shape
        N, C, H, W = x.shape

        out_h = (H + 2 * self.pad - FH) // self.stride + 1
        out_w = (W + 2 * self.pad - FW) // self.stride + 1

        col = im2col(x, FH, FW, self.stride, self.pad) # (N*out_H*out_W, C*FH*FW)
        flatten_filter = self.filter.reshape(FN, -1).T # (C*FH*FW, FN)
        out = np.dot(col, flatten_filter) + self.offset

        # ??? 为什么不能直接reshape成（N, -1, out_h, out_w）
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.flatten_filter = flatten_filter

        return out

    def backward(self, dout):
        '''
            dout.shape = (N, out_c, out_h, out_w)
        '''
        FN, C, FH, FW = self.filter.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)  # (-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout) # (C*FH*FW, N*out_H*out_W) * (-1, FN))
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.flatten_filter.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx

class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = (H - self.pool_h) // self.stride + 1
        out_w = (W - self.pool_w) // self.stride + 1

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        self.arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        self.x = x

        return out
        
    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size, ))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx

if __name__ == "__main__":
    pass