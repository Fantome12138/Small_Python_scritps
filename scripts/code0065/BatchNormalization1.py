import numpy as np
import torch
from torch import nn

'''https://zhuanlan.zhihu.com/p/100672008'''

class MyBN:
    
    def __init__(self, momentum, eps, num_features):
        """
        初始化参数值
        :param momentum: 追踪样本整体均值和方差的动量
        :param eps: 防止数值计算错误
        :param num_features: 特征数量
        """
        # 对每个batch的mean和var进行追踪统计
        self._running_mean = 0
        self._running_var = 1
        # 更新self._running_xxx时的动量
        self._momentum = momentum
        # 防止分母计算为0
        self._eps = eps
        # 对应论文中需要更新的beta和gamma，采用pytorch文档中的初始化值
        self._beta = np.zeros(shape=(num_features, ))
        self._gamma = np.ones(shape=(num_features, ))
        
    def batch_norm(self, x):
        """
        BN向传播
        :param x: 数据
        :return: BN输出
        """
        x_mean = x.mean(axis=0)
        x_var = x.var(axis=0)
        # 对应running_mean的更新公式, 仅测试时使用，训练时不用
        self._running_mean = (1-self._momentum)*x_mean + self._momentum*self._running_mean
        self._running_var = (1-self._momentum)*x_var + self._momentum*self._running_var
        # 对应论文中计算BN的公式
        x_hat = (x-x_mean)/np.sqrt(x_var+self._eps)
        y = self._gamma*x_hat + self._beta
        return y

data = np.array([[1, 2],
                 [1, 3],
                 [1, 4]]).astype(np.float32)

bn_torch = nn.BatchNorm1d(num_features=2)
bn_output_torch = bn_torch(torch.from_numpy(data))
print(bn_output_torch)

my_bn = MyBN(momentum=0.01, eps=0.001, num_features=2)
# Tensor.detach() 从计算图中脱离出来，返回一个新的tensor，新的tensor和原tensor共享数据内存
my_bn._beta = bn_torch.bias.detach().numpy() 
my_bn._gamma = bn_torch.weight.detach().numpy()
bn_output = my_bn.batch_norm(data, )
print(bn_output)