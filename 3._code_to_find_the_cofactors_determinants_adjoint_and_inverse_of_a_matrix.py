import numpy as np

def cofactor(A):
    m = A.shape[0]
    n = A.shape[1]
    cof = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            minor = np.delete(np.delete(A, i, 0), j, 1)
            cof[i][j] = (-1) ** (i + j) * np.linalg.det(minor)
    return cof

def det(A):
    return np.linalg.det(A)

def adjoint(A):
    return np.transpose(cofactor(A))

def inverse(A):
    return np.dot(np.linalg.inv(A), np.identity(A.shape[0]))

# Example usage
A = np.array([[1, 2], [3, 4]])
print("Cofactor Matrix:")
print(cofactor(A))
print("Determinant:")
print(det(A))
print("Adjoint Matrix:")
print(adjoint(A))
print("Inverse Matrix:")
print(inverse(A))
