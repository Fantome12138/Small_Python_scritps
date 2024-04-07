from matplotlib import pyplot as plt
import numpy as np
import math
import torch
import torch.nn as nn

'''sigmoid'''
def sigmoid_function(z):
    fz = []
    for num in z:
        fz.append(1/(1 + math.exp(-num)))
    return fz

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    """
    Args:
        x (numpy 数组): 输入数组。
    Returns:
        numpy 数组: 经过 Softmax 函数处理后的输出数组。
    """
    exp_x = np.exp(x - np.max(x))  # 减去最大值以避免数值不稳定
    return exp_x / np.sum(exp_x, axis=0)



def Tanh():
    m = nn.Tanh()

def relu():
    m = nn.RReLU(lower=0.1, upper=0.3)


class Swish(nn.Module):
    __constants__ = ['beta']
    beta: int
    def __init__(self, beta: int = 1) -> None:
        super(Swish, self).__init__()
        self.beta = beta
    def forward(self, input):
        return input*torch.sigmoid(input*self.beta)


class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        print("Mish avtivation loaded...")

    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x



if __name__ == '__main__':
    z = np.arange(-10, 10, 0.01)
    fz = sigmoid_function(z)
    plt.title('Sigmoid Function')
    plt.xlabel('z')
    plt.ylabel('σ(z)')
    plt.plot(z, fz)
    plt.show()
