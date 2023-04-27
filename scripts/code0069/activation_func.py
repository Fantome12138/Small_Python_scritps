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