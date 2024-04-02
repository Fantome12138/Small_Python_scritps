import numpy as np

matrix1 = [[1, 2], [3, 4]]
matrix2 = [[5, 6], [7, 8]]

# 计算矩阵之间的差
diff = [[a - b for a, b in zip(row1, row2)] for row1, row2 in zip(matrix1, matrix2)]
# 计算欧式距离
dist = sum(sum(row) ** 2 for row in diff) ** 0.5
print("欧式距离是：", dist)


matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])


dist = np.linalg.norm(matrix1 - matrix2)
print("欧式距离是：", dist)

# 计算两个矩阵的差的平方
squared_diff = (matrix1 - matrix2) ** 2
# 计算平方差的和
sum_squared_diff = np.sum(squared_diff)
# 计算欧式距离的平方根
european_distance = np.sqrt(sum_squared_diff)

print("欧式距离：", european_distance)
