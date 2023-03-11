import numpy as np
from model import CNNLSTM

batchsize = 32
datasize = (32, 64)   # 4s: 1024 points, 50% overlap, 0.25s window
channelsize = 22

embedding_dim = 16

x_shape = (batchsize, datasize[0], datasize[1], channelsize)    # (batchsize, 32, 64, 22)
y_shape = (batchsize, 1)

# take subject 1 as example
x_input_shape = (22, 1024*36497)
y_input_shape = (1, 1024*36497)

def splitdata(x, y, windowsize=1024):
    x = np.array(x)
    y = np.array(y)

    x = x.reshape(x_input_shape[0], -1, windowsize)
    x = x.transpose(1, 2, 0)    # (36497, 1024, 22)
    y = y.reshape(-1, windowsize)   # (36497, 1024)

    return x, y


