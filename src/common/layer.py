import numpy as np

class MulLayer:
    '''In_X * In_Y => Out'''
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    def backward(self, dout):
        '''
            para:
                - dout: the derivative of forward output.
            return:
                - dx : the partial derivative for x
                - dy : the partial derivative for y
        '''
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy

class AddLayer:
    ''' In_X + In_Y => Out'''
    def __init__(self):
        pass
    
    def forward(self, x, y):
        return x + y
    
    def backward(self, dout):
        '''
            The backwad propagation will directly transfer the derivative to from output to input
            para:
                - dout: the derivative of forward output.
            return:
                - dx : the partial derivative for x
                - dy : the partial derivative for y
        '''
        dx = dout * 1
        dy = dout * 1

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
        dx = np.dot(dout * self.w.T)
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
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx


if __name__ == "__main__":
    pass