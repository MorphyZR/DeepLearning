import numpy as np 
def step_func(x: np.array) -> np.array:
        '''
            step(x) = if x > 0 return 1 else return 0
        '''
        mask1 = x <= 0
        mask2 = x > 0
        x[mask1] = 0
        x[mask2] = 1
        return x

def sigmoid(x: np.array) -> np.array:
    '''
        y = 1 / (1 + exp(-x))
    '''
    return 1 / (1 + np.exp(-x))

def relu(x: np.array) -> np.array:
    '''
        y = max(x, 0)
    '''
    mask = x <= 0
    x[mask] = 0
    return x

def identity(x):
        return x

def softmax_classic(input: np.array) -> np.array:
    '''
        out[i] = exp(input[i]) / sum(exp(input)) 
        Warning: this function has the risk of overflow
    '''
    exp_a = np.exp(input)
    sum_exp = np.sum(exp_a)
    output = exp_a / sum_exp
    return output

# TODO
def softmax(x: np.array) -> np.array:
    '''
        out[i] = exp(input[i] - max(a)) / sum(exp(input))
    '''
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

    # c = np.max(x,axis=1).reshape(x.shape[0], -1)
    # exp_a = np.exp(x - c)
    # sum_exp = np.sum(exp_a, axis=1).reshape(x.shape[0], -1)
    # output = exp_a / sum_exp
    # return output

def mean_squared_error(x: np.array, t: np.array) -> float:
    '''
        loss = 0.5 * sum((x[i] - t[i])^2)
    '''
    return 0.5 * np.sum(pow(x - t, 2))

def cross_entropy_error(y: np.array, t: np.array) -> float:
    '''
        loss = - SUM(t[i] * log(x[i]))
    '''
    # reshape 1-d vector to (1,n) matrix
    # if y.ndim == 1:
    #     t = t.reshape(1, t.size)
    #     y = y.reshape(1, y.size)
    # batch_size = y.shape[0]
    # delta = 1e-7
    # return -np.sum(t * np.log(y + delta)) / batch_size
    # return -np.sum(np.log(x[np.arange(batch_size), t] + delta)) / batch_size
    
    # reshape 1-d vector to (1,n) matrix
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # If training data is one-hot-vectorm convert to correct index
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    # only need to calculate the true label
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def _numerical_gradient_no_batch(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val 
        
    return grad

def numerical_gradient(f, x: np.array) -> np.array:
    '''
        f is the function
    '''
    if x.ndim == 1:
        return _numerical_gradient_no_batch(f, x)
    else:
        grad = np.zeros_like(x)
        for i, row_x in enumerate(x):
            grad[i] = _numerical_gradient_no_batch(f, row_x)
        return grad

def change_to_one_hot(x: np.array, num_label: int) -> np.array:
    one_hot = np.zeros((x.size, num_label), dtype=int)
    for i, row in enumerate(x):
        one_hot[i, x[i]] = 1
        # row[x[i]] = 1
    return one_hot


if __name__ == "__main__":
    x = np.array([
        [0.6, 0.1, 0.3],
        [0.2, 0.1, 0.7], 
    ])
    t = np.array([
        [1,0,0],
        [0,0,1]
    ])
    loss = mean_squared_error(x, t)
    print(loss)
    print(cross_entropy_error(x, t))

    foo = lambda x : x[0]**2 + x[1]**2
    print(numerical_gradient(foo, np.array([3.0, 4.0])))