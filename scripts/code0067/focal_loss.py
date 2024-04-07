import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def dice_loss(y_true, y_pred, smooth=1):
    """
    - y_true: 真实标签的独热编码形式，形状应与 y_pred 相同
    - y_pred: 模型的预测分布
    - smooth: 一个很小的数，用于防止分母为零
    """
    intersection = np.sum(y_true * y_pred, axis=(1, 2, 3))  # 沿着通道和其他维度求和
    union = np.sum(y_true, axis=(1, 2, 3)) + np.sum(y_pred, axis=(1, 2, 3))  # 沿着通道和其他维度求和
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def CE_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def BCE_loss(pred, label, weights):
    loss = weights * (label * np.log(pred) + (1 - label) * np.log(1 - pred))
    return -loss

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    '''
    Args:
        y_true (np.ndarray): True labels (0 or 1).
        y_pred (np.ndarray): Predicted probabilities (between 0 and 1).
        gamma (float): Focal loss hyperparameter (default: 2.0).
        alpha (float): Focal loss hyperparameter (default: 0.25).
    '''
    # 对预测概率值 y_pred 进行截断，以避免数值不稳定性
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    loss = - (alpha * y_true + (1 - y_true)) * \
            (1 - y_pred) ** gamma * np.log(y_pred)
    return np.mean(loss)



y_true = np.array([0, 1., 1., 0])
y_pred = np.array([-3., 8., 2, 9])
sigmoid_pred = sigmoid(y_pred)
print(sigmoid_pred) # array([0.04742587, 0.99966465, 0.88079708, 0.99987661]) 经过sigmoid，只有-3是小于0.5的
print(CE_loss(sigmoid_pred, y_true))

## numpy 实现CE结果
print(CE_loss(sigmoid_pred, y_true).sum() / len(y_true))

y_true_t = torch.from_numpy(y_true)
y_pred_t = torch.from_numpy(y_pred)
sigmoid_pred_t = torch.from_numpy(sigmoid_pred)
## torch 实现CE结果
print(F.binary_cross_entropy_with_logits(y_pred_t, y_true_t))

## numpy 实现BCE结果
weights = np.array([1, 2, 2, 1])
print(BCE_loss(sigmoid_pred, y_true, weights).sum() / len(y_true))

## torch 实现BCE结果
weights_t = torch.from_numpy(weights)
loss = nn.BCELoss(weights_t)
l2 = loss(sigmoid_pred_t, y_true_t)
print(l2)

## numpy 实现FC结果
print(focal_loss(sigmoid_pred, y_true, weights).sum() / len(y_true))
