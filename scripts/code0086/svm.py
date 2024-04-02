import numpy as np

class LinearSVM:
    def __init__(self, learning_rate=0.0001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.w = None  # 权重向量
        self.b = None  # 偏置项

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        # 梯度下降
        for _ in range(self.n_iterations):
            for i in range(n_samples):
                # 计算预测值和间隔
                y_pred = np.dot(self.w, X[i]) + self.b
                margin = y_pred - y[i]
                # 更新权重和偏置
                if y_pred * y[i] <= 1:  # 如果在间隔边界内或错误分类
                    # 计算梯度
                    gradient_w = -self.learning_rate * (2 * X[i] * y[i] - 2 * np.sum(X * self.w))
                    gradient_b = -self.learning_rate * (2 * y[i])
                    # 更新参数
                    self.w -= gradient_w
                    self.b -= gradient_b

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)


X = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 1], [3, 2]])
y = np.array([1, 1, 1, -1, -1, -1])
svm = LinearSVM(learning_rate=0.0001, n_iterations=1000)
svm.fit(X, y)
predictions = svm.predict(X)
print("Predictions:", predictions)
