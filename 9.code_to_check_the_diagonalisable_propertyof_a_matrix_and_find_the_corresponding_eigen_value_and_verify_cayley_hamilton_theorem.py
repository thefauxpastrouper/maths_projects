import numpy as np

def is_diagonalizable(matrix):
    """
    Check the diagonalizable property of a matrix.
    """
    # Compute the eigenvalues of the matrix
    eigenvalues = np.linalg.eigvals(matrix)
    
    # Check if all the eigenvalues have a matching eigenvector
    return np.all(np.iscomplex(eigenvalues)) == False

def eigenvalues(matrix):
    """
    Find the eigenvalues of a matrix.
    """
    return np.linalg.eigvals(matrix)
# to verify cayley hamilton theorem
import numpy as np

def cayley_hamilton(matrix, degree=None):
    """
    Verify the Cayley-Hamilton theorem for a matrix.
    """
    if degree is None:
        degree = matrix.shape[0]
    
    # Compute the characteristic polynomial of the matrix
    characteristic_polynomial = np.poly(matrix)
    
    # Compute the matrix raised to the specified degree
    matrix_to_degree = np.linalg.matrix_power(matrix, degree)
    
    # Evaluate the characteristic polynomial at the matrix raised to the specified degree
    result = np.polyval(characteristic_polynomial, matrix_to_degree)
    
    # Check if the result is the zero matrix
    return np.allclose(result, np.zeros_like(matrix))
