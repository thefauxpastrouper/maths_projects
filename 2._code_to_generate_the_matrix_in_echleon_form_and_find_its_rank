import numpy as np

def gaussian_elimination(A):
    n, m = A.shape
    for i in range(n):
        # Find the pivot row
        pivot = np.argmax(np.abs(A[i:, i])) + i
        # Swap the pivot row with the current row
        A[[i, pivot]] = A[[pivot, i]]
        # Divide the current row by the pivot element
        A[i] //= A[i, i]
        # Eliminate the pivot element from the other rows
        for j in range(n):
            if i != j:
                A[j] -= A[j, i] * A[i]
    return A

# Example usage
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(A)
A_echelon = gaussian_elimination(A)
rank = np.linalg.matrix_rank(A_echelon)
print(A_echelon)
print("rank=",rank)
