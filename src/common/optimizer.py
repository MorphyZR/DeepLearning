import numpy as np
class Optimizer:
    def __init__(self, lr: float):
        '''
        @parms:
            - lr : learning rate
        '''
        self.lr = lr

    def update(self, params, grads):
        '''
        @para
            - params : [dict] the weight of network
            - grads : [dict] gradient fo weights
        '''
        pass

class SGD(Optimizer):
    '''
        stochastic gradient descent
        W <- W - lr * (dL/dW)
    '''

    def update(self, params, grads) -> dict:
        '''
        upadte the params;
        '''
        for k, v in grads.items():
            params[k] -= self.lr * v
        
   

class Momentum(Optimizer):
    '''
    v <- av - lr * (dL / dW)
    W <- W + v

    - a: momentum; usually choos value like 0.9
    - lr: learning rate
    - dL / dW: gradient
    - W: parameters
    '''
    def __init__(self, lr, momentum=0.9):
        super().__init__(lr)
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for k, v in params.items():
                self.v[k] = np.zeros_like(v, dtype=float)

        for k in grads.keys():
            self.v[k] = self.momentum * self.v[k] - self.lr * grads[k]
            params[k] += self.v[k] 

class AdaGrad(Optimizer):
    '''
    AdaGrad optimizer
    h <- h + (dL/dW) ** 2  
    W <- W - lr * (dL/dW) / sqrt(h)
    '''
    def __init__(self, lr):
        super().__init__(lr)
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for k, v in params.items():
                self.h[k] = np.zeros_like(v)
        
        for k in params.keys():
            self.h[k] += grads[k] * grads[k]
            # add a extreme value 1e-7 to avoid divid zero error
            params[k] -= self.lr * grads[k] / (np.sqrt(self.h[k]) + 1e-7)

class Adam(Optimizer):
    '''
    Adam Optimizer
    t <- t + 1
    m <- beta1 * m + (1 - beta1) * (dL/dW)
    v <- beta2 * v + (1 - beta2) * ((dL/dW)**2)
    unbias_m <- m / (1 - beta1 ** t)
    unbias_v <- v / (1 - beta2 ** t)
    W <- W - lr * unbias_m / sqrt(unbias_v)
    '''
    def __init__(self, lr, beta_1=0.9, beta_2=0.999):
        super().__init__(lr)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.iter = 0
        self.m = None # first-order moment estimate
        self.v = None # first-order moment estimate

    def update(self, params, grads):
        self.iter += 1
        if self.m is None:
            self.m, self.v = {}, {}
            for k in params.keys():
                self.m[k] = np.zeros_like(params[k])
                self.v[k] = np.zeros_like(params[k])
        
        lr_t = self.lr * np.sqrt(1 - self.beta_2 ** self.iter) / (1 - self.beta_1 ** self.iter)
        
        for k in params.keys():
            self.m[k] = self.beta_1 * self.m[k] + (1 - self.beta_1) * grads[k]
            self.v[k] = self.beta_2 * self.v[k] + (1 - self.beta_2) * grads[k] * grads[k]
            # correct bias
            # unbias_m = self.m[k] / (1.0 - (self.beta_1 ** self.iter)) 
            # unbias_v = self.v[k] / (1.0 - (self.beta_2 ** self.iter))
            # params[k] -= self.lr * unbias_m / (np.sqrt(unbias_v) + 1e-7)
            params[k] -= lr_t * self.m[k] / (np.sqrt(self.v[k]) + 1e-7)
            