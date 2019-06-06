import numpy as np
from layers.simple_layer import MulLayer, AddLayer

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
        
if __name__ == "__main__":
    x = np.array([[1, -2], [2, -3]])
    relu = Relu()
    out = relu.forward(x)
    print(out)
    print(relu.backward(out))
    print(relu.mask)