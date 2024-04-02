import numpy as np

def svd(A, full_matrices=True):
    # 计算A的转置乘以A
    A_trans_A = np.dot(A.T, A)
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(A_trans_A)
    # 排序特征值和特征向量，使得最大的特征值在最前面
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # 计算奇异值，它们是特征值的平方根
    sigma = np.sqrt(eigenvalues)
    # 计算V
    V = eigenvectors
    # 计算U
    if full_matrices:
        U = np.dot(A, V) / sigma
    else:
        U = np.dot(A, V)
    return U, sigma, V

# 示例矩阵
A = np.array([[1, 2], [3, 4], [5, 6]])
U, sigma, V = svd(A)
print("U:\n", U)
print("Sigma:\n", sigma)
print("V:\n", V)
