'''
numpy 实现
'''
import numpy as np
# 定义输入序列a和卷积核v
a = [1, 2, 3]
v = [0, 1, 0.5]

# 计算线性卷积
linear_convolution = np.convolve(a, v, mode='full')
print("线性卷积结果：", linear_convolution)
# 计算互相关
cross_correlation = np.correlate(a, v, mode='full')
print("互相关结果：", cross_correlation)
# 计算自相关
auto_correlation = np.correlate(a, a, mode='full')
print("自相关结果：", auto_correlation)


'''
python 实现
'''
def linear_convolution(a, v):
    # 初始化结果数组
    result = [0] * (len(a) + len(v) - 1)
    # 计算线性卷积
    for i in range(len(a)):
        for j in range(len(v)):
            result[i + j] += a[i] * v[j]
    return result

def cross_correlation(a, v):
    # 初始化结果数组
    result = [0] * (len(a) + len(v) - 1)
    v = v[::-1]
    # 计算互相关
    for i in range(len(a)):
        for j in range(len(v)):
            result[i + j] += a[i] * v[j]
    return result

def auto_correlation(a):
    # 初始化结果数组
    result = [0] * (2 * len(a) - 1)
    v = a[::-1]
    # 计算自相关
    for i in range(len(a)):
        for j in range(len(a)):
            result[i + j] += a[i] * v[j]
    return result

# 定义输入序列a和卷积核v
a = [1, 2, 3]
v = [0, 1, 0.5]

# 计算线性卷积
print("线性卷积结果：", linear_convolution(a, v))
# 计算互相关
print("互相关结果：", cross_correlation(a, v))
# 计算自相关
print("自相关结果：", auto_correlation(a))
