import numpy as np

def gram_schmidt_orthogonalization(vectors):
    """
    Find the orthonormal basis of a given vector space using the Gram-Schmidt orthogonalization process.
    """
    orthonormal_basis = []
    
    for i, vector in enumerate(vectors):
        # Subtract the projections of the previous orthonormal basis vectors onto the current vector
        for u in orthonormal_basis:
            vector = vector - np.dot(vector, u) * u
        
        # Normalize the current vector to obtain an orthonormal basis vector
        u = vector / np.linalg.norm(vector)
        
        orthonormal_basis.append(u)
    
    return orthonormal_basis
