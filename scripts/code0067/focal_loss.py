import numpy as np
import torch
import torch.nn.functional as F 
from torch import nn

def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def CE(pred, label):
    loss = label * np.log(pred) + (1 - label) * np.log(1 - pred)
    return -loss

def my_BCE(pred, label, weights):
    loss = weights * (label * np.log(pred) + (1 - label) * np.log(1 - pred))
    return -loss

def my_focal(logits, label, gamma=2, weights=0.25):
    loss = weights * (label * np.log(logits) * (1 - logits) ** gamma + (1 - label) * np.log(1 - logits) * (logits) ** gamma)
    return -loss

y_true = np.array([0, 1., 1., 0])
y_pred = np.array([-3., 8., 2, 9])
sigmoid_pred = sigmoid(y_pred)
print(sigmoid_pred) # array([0.04742587, 0.99966465, 0.88079708, 0.99987661]) 经过sigmoid，只有-3是小于0.5的
print(CE(sigmoid_pred, y_true))

## numpy 实现CE结果
print(CE(sigmoid_pred, y_true).sum() / len(y_true))

y_true_t = torch.from_numpy(y_true)
y_pred_t = torch.from_numpy(y_pred)
sigmoid_pred_t = torch.from_numpy(sigmoid_pred)
## torch 实现CE结果
print(F.binary_cross_entropy_with_logits(y_pred_t, y_true_t))

## numpy 实现BCE结果
weights = np.array([1, 2, 2, 1])
print(my_BCE(sigmoid_pred, y_true, weights).sum() / len(y_true))

## torch 实现BCE结果
weights_t = torch.from_numpy(weights)
loss = nn.BCELoss(weights_t)
l2 = loss(sigmoid_pred_t, y_true_t)
print(l2)

## numpy 实现FC结果
print(my_focal(sigmoid_pred, y_true, weights).sum() / len(y_true))


