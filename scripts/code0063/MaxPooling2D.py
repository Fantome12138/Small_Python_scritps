import numpy as np
import torch


class MaxPooling2D:
    def __init__(self, kernel_size=(2, 2), stride=2):
        self.kernel_size = kernel_size
        self.w_height = kernel_size[0]
        self.w_width = kernel_size[1]
        self.stride = stride

        self.x = None
        self.in_height = None
        self.in_width = None

        self.out_height = None
        self.out_width = None

        self.arg_max = None

    def __call__(self, x):
        self.x = x
        self.in_height = np.shape(x)[0]
        self.in_width = np.shape(x)[1]

        self.out_height = int((self.in_height - self.w_height) / self.stride) + 1
        self.out_width = int((self.in_width - self.w_width) / self.stride) + 1

        out = np.zeros((self.out_height, self.out_width))
        self.arg_max = np.zeros_like(out, dtype=np.int32)

        for i in range(self.out_height):
            for j in range(self.out_width):
                start_i = i * self.stride
                start_j = j * self.stride
                end_i = start_i + self.w_height
                end_j = start_j + self.w_width
                print(f'{start_i} {start_j} : {end_i} {end_j}')
                out[i, j] = np.max(x[start_i: end_i, start_j: end_j])
                # 放置原数值的索引
                self.arg_max[i, j] = np.argmax(x[start_i: end_i, start_j: end_j])  
        return out

    def backward(self, d_loss):
        dx = np.zeros_like(self.x)
        for i in range(self.out_height):
            for j in range(self.out_width):
                start_i = i * self.stride
                start_j = j * self.stride
                end_i = start_i + self.w_height
                end_j = start_j + self.w_width
                # 求出数组某元素（或某组元素）拉成一维后的索引值在原本维度（或指定新维度）中对应的索引
                index = np.unravel_index(self.arg_max[i, j], self.kernel_size)
                dx[start_i:end_i, start_j:end_j][index] = d_loss[i, j] #
        return dx

# 用于控制Python中小数的显示精度      
np.set_printoptions(precision=8, suppress=True, linewidth=120)

x_numpy = np.random.random((1, 1, 6, 9))
# requires_grad=True 为这个张量计算梯度
x_tensor = torch.tensor(x_numpy, requires_grad=True)  

max_pool_numpy = MaxPooling2D((2, 2), stride=2)
max_pool_tensor = torch.nn.MaxPool2d((2, 2), 2)

out_numpy = max_pool_numpy(x_numpy[0, 0]) # -> shape (3,4)
out_tensor = max_pool_tensor(x_tensor)

d_loss_numpy = np.random.random(out_tensor.shape)
d_loss_tensor = torch.tensor(d_loss_numpy, requires_grad=True)
out_tensor.backward(d_loss_tensor)

dx_numpy = max_pool_numpy.backward(d_loss_numpy[0, 0])
dx_tensor = x_tensor.grad
# print("out_numpy: \n", out_numpy)
print("out_tensor \n", out_tensor.data.numpy())
# print("dx_numpy: \n", dx_numpy)
print("dx_tensor \n", dx_tensor.data.numpy())            


