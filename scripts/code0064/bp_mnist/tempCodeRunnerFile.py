import random
import numpy as np

def load_batches(data, batch_size):

    n = len(data[0])
    # 对数据进行洗牌
    shuffle_idx = random.sample(range(n), n)
    print(shuffle_idx, type(shuffle_idx), data[0])
    X = data[0][shuffle_idx]

    return X


data = np.array([[1,2,3,4,5,6,7,8],[0,0,1,0,1,0,0,0]])
batch_size = 2
x = load_batches(data, batch_size)
print(x )


sizes = [4,4,2]
dws = [np.zeros((i, j)) for i, j in zip(sizes[:-1], sizes[1:])]
print('###', dws, len(dws))