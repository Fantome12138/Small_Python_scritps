import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cross_entropy_loss(y_true, y_pred):
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

class LogisticRegression:
    def __init__(self, num_features):
        self.weights = np.zeros((num_features, 1))
        self.bias = 0

    def fit(self, X, y, learning_rate=0.01, num_iterations=1000):
        for i in range(num_iterations):
            y_pred = sigmoid(np.dot(X, self.weights) + self.bias)
            loss = np.mean(cross_entropy_loss(y, y_pred))
            # 计算梯度
            dW = np.dot(X.T, (y_pred - y)) / len(y)
            db = np.mean(y_pred - y)
            # 更新权重参数
            self.weights -= learning_rate * dW
            self.bias -= learning_rate * db
            # 打印损失值
            if i % 100 == 0:
                print(f"Iteration {i}, loss: {loss:.4f}")

    def predict(self, X):
        y_pred = sigmoid(np.dot(X, self.weights) + self.bias)
        return np.round(y_pred)

X = np.array([[2.0, 70.0, 120.0],
              [3.0, 80.0, 130.0],
              [4.0, 90.0, 140.0],
              [5.0, 100.0, 150.0]])
y = np.array([0, 0, 1, 1]).reshape(-1, 1)

model = LogisticRegression(num_features=3)
model.fit(X, y, learning_rate=0.1, num_iterations=1000)
X_new = np.array([[6.0, 110.0, 160.0]])
y_pred = model.predict(X_new)
print(f"Predicted label for X_new: {y_pred}")
