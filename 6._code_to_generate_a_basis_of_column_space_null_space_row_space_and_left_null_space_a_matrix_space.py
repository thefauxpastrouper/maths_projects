import numpy as np

def basis_of_subspace(A):
    # Compute the SVD decomposition of the matrix
    U, s, Vh = np.linalg.svd(A, full_matrices=False)

    # Column space
    column_space = U[:, :len(s)]
    
    # Null space
    null_space = Vh[len(s):, :].T
    
    # Row space
    row_space = U[:, :len(s)].dot(np.diag(s))
    
    # Left null space
    left_null_space = Vh[:len(s), :].T
    
    return column_space, null_space, row_space, left_null_space
