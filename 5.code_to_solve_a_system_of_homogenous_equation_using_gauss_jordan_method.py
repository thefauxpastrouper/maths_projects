def gauss_jordan(A):
    n = len(A)
    for i in range(n):
        # find pivot
        pivot = i
        for j in range(i+1, n):
            if abs(A[j][i]) > abs(A[pivot][i]):
                pivot = j
                
        # swap rows
        if pivot != i:
            A[pivot], A[i] = A[i], A[pivot]
            
        # divide the pivot row by the pivot element
        pivot_value = A[i][i]
        for j in range(i, n+1):
            A[i][j] /= pivot_value
            
        # eliminate elements in pivot column
        for j in range(n):
            if j != i:
                factor = A[j][i]
                for k in range(i, n+1):
                    A[j][k] -= factor * A[i][k]
    return A
