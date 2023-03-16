import numpy as np
from module import Layers

class BatchNormlization(Layers):
    """
    https://zhuanlan.zhihu.com/p/483888908
    https://blog.csdn.net/weixin_44754861/article/details/108343938?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_utm_term~default-5.pc_relevant_paycolumn_v3&spm=1001.2101.3001.4242.4&utm_relevant_index=7

    """
    def __init__(self, name, x,eps =1e-7, momentum =0.9, mode = "train"):
        super(BatchNormlization).__init__(name)
        self.eps =eps
        self.input = x
        n, c, h, w = x.shape
        self.momentum = momentum
        self.running_mean = np.zeros(c)
        self.running_var = np.zeros(c)
        self.gamma = np.random(c)
        self.beta =np.random(c)
        self.mode = mode

    def add_dim(x, dim):
        return np.expand_dims(x, axis=dim) # batch 

    def forward(self):
        ib, ic, ih, iw = self.input.shape

        self.input = self.input.transpose(1, 0, 2, 3).reshape([ic, -1]) # n,c,h,w ->c, n*h*w
        if self.mode == "train":
            self.var = np.sqrt(self.var +self.eps) # 
            self.mean = np.mean(self.input, axis=0) # 每个channel的均值
            self.mean = self.add_dim(self.mean, 1) # 与后面的self.input 维度一致
            self.var = np.var(self.input, axis=0) #每个channel的方差
            self.var = self.add_dim(self.var , 1)
            self.gamma = self.add_dim(self.gamma, 1)
            self.beta = self.add_dim(self.beta, 1)
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) *self.mean
            self.running_var = self.momentum * self.running_var + (1-self.momentum) *self.var
            self.input_ = (self.input -  self.running_mean)/(self.running_var + self.eps)
            dout = (self.input_*self.gamma +self.beta ).reshape(ic,ib, ih, iw).transpose(1, 0, 2, 3)
            self.cache = (self.input_, self.gamma, (self.input - self.running_mean, self.running_var + self.eps))
        elif self.mode == "test":
            x_hat = (self.input - self.running_mean) / (np.sqrt(self.running_var + self.eps))
            dout = self.gamma * x_hat + self.beta
        else:
            raise ValueError("Invalid forward batch normlization mode")
        return dout, self.cache


    def backward(self, dout):
        N, D = dout.shape
        x_, gamma, x_minus_mean, var_plus_eps =self.cache

        # calculate gradients
        dgamma = np.sum(x_ * dout, axis=0)
        dbeta = np.sum(dout, axis=0)

        dx_ = np.matmul(np.ones((N,1)), gamma.reshape((1, -1))) * dout
        dx = N * dx_ - np.sum(dx_, axis=0) - x_ * np.sum(dx_ * x_, axis=0)
        dx *= (1.0/N) / np.sqrt(var_plus_eps)

        return dx, dgamma, dbeta

    def update(self, lr, dgamma, dbeta):
        self.gamma -= dgamma *lr
        self.beta -= dbeta*lr