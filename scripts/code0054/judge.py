"""
@author: hukai
@Time: 2022/4/6 15:04

confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值，与之前介绍的相反
P\L     P    N
P      TP    FP
N      FN    TN
"""
import numpy as np
class SegmentationMetric(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusionMatrix = np.zeros((self.num_classes, self.num_classes))
        
    def updateConfusionMatrix(self, predict, label):
        mask = (label >= 0) & (label < self.num_classes)  # 寻找target中为目标的像素索引
        new_label = self.num_classes * label[mask] + predict[mask]  # 二维的predict拉到一维
        count = np.bincount(new_label, minlength=self.num_classes**2)
        self.confusionMatrix += count.reshape(self.num_classes, self.num_classes)
        return self.confusionMatrix

    def pixelAccuracy(self):
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # 横着代表预测值，竖着代表真实值
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  # 求平均值，遇到Nan类型，其值取为0
        return meanAcc

    def meanIOU(self):
        intersection = np.diag(self.confusionMatrix)  # 对角线元素
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        IOU = intersection / union
        mIOU = np.nanmean(IOU)
        return mIOU

if __name__ == '__main__':
    Predict = np.array([[0, 0, 1, 0, 2, 2], [0, 1, 1, 1, 2, 2]])  # 可直接换成预测图片
    Label = np.array([[0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2]])  # 可直接换成标注图片
    metric = SegmentationMetric(3)  # 3表示有3个分类，有几个分类就填几
    confusionMatrix = metric.updateConfusionMatrix(Predict, Label)
    pa = metric.pixelAccuracy()
    cpa = metric.classPixelAccuracy()
    mpa = metric.meanPixelAccuracy()
    mIoU = metric.meanIOU()
    print(confusionMatrix)
    print('pa is : %f' % pa)
    print('cpa is :') # 列表
    print(cpa)
    print('mpa is : %f' % mpa)
    print('mIoU is : %f' % mIoU)
