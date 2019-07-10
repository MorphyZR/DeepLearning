import numpy as np
def im2col(input_data, fh: int, fw: int, stride=1, pad=0):
    '''
        convert 3D image matrix to 1D column
    @params
        - input_data: the input images, (N, C, H, W);
        - fh: height of filter
        - fw: width of filter
        - stride:
        - pad

    @return
        2-D array, (N*out_h*out_w, C * fh * fw)
        Each row is a flatten data
    '''
    # padding
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')

    N, C, H, W = img.shape
    out_h = (H - fh) // stride + 1
    out_w = (W - fw) // stride + 1

    # col = np.zeros((N, out_h, out_w, C, fh, fw))
    # for y in range(out_h):
    #     for x in range(out_w):
    #         col[:, y, x] = img[
    #             ..., y*stride : y*stride+fh, x*stride : x*stride+fw
    #             ]
    # return col.reshape(np.multiply.reduceat(col.shape, (0, 3)))

    # Strides是遍历数组时每个维度中需要步进的字节数
    NN, CC, HH, WW = img.strides
    # breakpoint()
    col = np.lib.stride_tricks.as_strided(
            img, 
            shape=(N, out_h, out_w, C, fh, fw), 
            strides=(NN, stride * HH, stride * WW, CC, HH, WW)
        ).astype(float)
    return col.reshape(np.multiply.reduceat(col.shape, (0, 3)))

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 入力データの形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

