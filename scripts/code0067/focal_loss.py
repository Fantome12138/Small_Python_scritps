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



##############################################################

from torch import nn
import torch
from torch.nn import functional as F

class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print(" --- Focal_loss alpha = {}, 将对每一类权重进行精细化赋值 --- ".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]，为 one-hot 编码格式
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
