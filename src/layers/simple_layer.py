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