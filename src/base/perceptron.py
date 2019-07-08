from typing import List
import numpy as np

class Perceptron:
    '''
        AND, NAND, OR have the same structure, only the weight and offset is different.
    '''

    def AND(x1: int, x2: int) -> int:
        '''
            if x1 * w1 + x2 * w2 <= b, then return 0
            else return 1
        '''
        x = np.array([x1, x2])
        w = np.array([0.5, 0.5])
        b = -0.7
        tmp = np.dot(x, w) + b
        if tmp <= 0:
            return 0
        else:
            return 1

    def NAND(x1: int, x2: int) -> int:
        '''
            if x1 * w1 + x2 * w2 <= b, then return 1
            else return 0
        '''
        x = np.array([x1, x2])
        w = -np.array([0.5, 0.5])
        b = 0.7

        tmp = np.dot(x, w) + b
        if tmp <= 0:
            return 0
        else:
            return 1

    def OR(x1: int, x2: int) -> int:
        x = np.array([x1, x2])

        w = np.array([0.5, 0.5])
        b = -0.2

        tmp = np.dot(x, w) + b
        if tmp <= 0:
            return 0
        else:
            return 1

    def XOR(x1: int, x2: int) -> int:
        '''
            XOR(x1, x2) = AND( OR(x1, x2), NAND(x1, x2))
        '''
        y1 = Perceptron.OR(x1, x2)
        y2 = Perceptron.NAND(x1, x2)
        return Perceptron.AND(y1, y2)


if __name__ == "__main__":
    arr_1 = 1
    arr_2 = 0
    print(Perceptron.AND(arr_1, arr_2))
    print(Perceptron.NAND(arr_1, arr_2))
    print(Perceptron.OR(arr_1, arr_2))
    print(Perceptron.XOR(arr_1, arr_2))