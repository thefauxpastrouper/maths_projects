import numpy as np

def gauss_elimination(A, b):
    n = len(b)
    for i in range(n):
        max_index = np.argmax(np.abs(A[i:, i])) + i
        if i != max_index:
            A[[i, max_index]] = A[[max_index, i]]
            b[[i, max_index]] = b[[max_index, i]]
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] -= factor * A[i][k]
            b[j] -= factor * b[i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i][i]
    return x

# Example usage
A = np.array([[1, 2, 1], [3, 8, 1], [0, 4, 1]])
b = np.array([3, 9, 4])
print("Solution:")
print(gauss_elimination(A, b))
