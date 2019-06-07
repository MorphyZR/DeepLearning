class SGD:
    '''
        stochastic gradient descent
        W <- W - lr * (dL/dW)
    '''
    def __init__(self, lr=0.01):
        '''
        @para:
            - lr    : [float], learning rate
        '''
        self.lr = lr

    def update(self, params, grads):
        pass

class Momentum: