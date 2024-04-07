import autograd.numpy as np
from autograd import grad
'''
autograd以一种非常直观和方便的方式来定义和计算多变量函数的梯度。autograd库通过装饰器grad来实现这一点
'''

class Adam:
    def __init__(self, loss, weights, lr=0.001, beta1=0.9, beta2=0.999, epislon=1e-8):
        self.loss = loss
        self.theta = weights
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epislon = epislon
        self.get_gradient = grad(loss)
        self.m = 0
        self.v = 0
        self.t = 0

    def minimize_raw(self):
        self.t += 1
        g = self.get_gradient(self.loss)
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * (g * g)
        self.m_hat = self.m / (1 - self.beta1 ** self.t)
        self.v_hat = self.v / (1 - self.beta2 ** self.t)
        self.theta = self.theta - self.lr * self.m_hat / (self.v_hat ** 0.5 + self.epislon)


class AdamW:
    def __init__(self, loss, weights, lambda1, lr=0.001, beta1=0.9, beta2=0.999, epislon=1e-8):
        self.loss = loss
        self.theta = weights
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epislon = epislon
        self.lambda1 = lambda1
        self.get_gradient = grad(loss)
        self.m = 0
        self.v = 0
        self.t = 0

    def minimize_raw(self):
        self.t += 1
        g = self.get_gradient(self.loss)
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * (g * g)
        self.m_hat = self.m / (1 - self.beta1 ** self.t)
        self.v_hat = self.v / (1 - self.beta2 ** self.t)
        self.theta = self.theta - self.lr * (
                    self.m_hat / (self.v_hat ** 0.5 + self.epislon) + self.lambda1 * self.theta)


import numpy as np

def sgd(X, y, learning_rate=0.01, epochs=100, batch_size=1):
    """
    参数:
    X (numpy array): 输入数据，形状=(n_samples, n_features)
    y (numpy array): 目标值，形状=(n_samples,)
    learning_rate (float): SGD的学习率
    epochs (int): SGD的迭代次数
    batch_size (int): SGD的批量大小
    返回:
    w (numpy array): 学习到的权重，形状=(n_features,)
    b (float): 学习到的偏置
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    for _ in range(epochs):
        for i in range(0, n_samples, batch_size):
            X_i = X[i:i+batch_size]
            y_i = y[i:i+batch_size]

            y_pred = np.dot(X_i, w) + b
            grad_w = (1/batch_size) * np.dot(X_i.T, (y_pred - y_i))
            grad_b = (1/batch_size) * np.sum(y_pred - y_i)

            w -= learning_rate * grad_w
            b -= learning_rate * grad_b

    return w, b
