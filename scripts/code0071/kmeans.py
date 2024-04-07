import numpy as np

def knn(train_set, train_labels, test_instance, k):
    """
    - train_set: 训练集，形状为 (num_samples, num_features)
    - train_labels: 训练集的标签，形状为 (num_samples,)
    - test_instance: 测试实例，形状为 (num_features,)
    - k: 选取的最近邻居的数量
    - 预测的标签
    """
    # 计算测试实例与训练集中每个实例的距离
    distances = np.sqrt(np.sum((train_set - test_instance)**2, axis=1))
    # 获取距离最小的k个实例的索引
    k_nearest_neighbors = np.argsort(distances)[:k]
    # 获取这k个实例的标签
    k_nearest_labels = train_labels[k_nearest_neighbors]
    # 返回最常见的标签
    c = np.bincount(k_nearest_labels)
    most_common_label = np.argmax(c)
    return most_common_label

X_train = np.array([
    [2.1, 1.3],
    [1.3, 3.2],
    [2.9, 2.5],
    [2.5, 2.3],
    [4.6, 2.9],
    [3.8, 1.9],
    [3.1, 1.5]
])
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])
# 测试数据
X_test = np.array([[3.2, 2.8]])
predictions = knn(X_train, y_train, X_test, k=3)
print("预测结果:", predictions)
