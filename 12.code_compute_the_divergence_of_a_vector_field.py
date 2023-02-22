import numpy as np

def divergence(vector_field, step_size=0.01):
    """
    Compute the divergence of a vector field.
    """
    # Get the shape of the vector field
    shape = np.array(vector_field.shape[:-1])
    
    # Create a grid of points
    grid = np.array(np.meshgrid(*[np.arange(i) for i in shape], indexing='ij')).astype(float)
    
    # Compute the divergence of the vector field by finite differences
    divergence = np.zeros(shape)
    for i in range(len(grid)):
        grid_plus = grid.copy()
        grid_plus[i, :] += step_size
        grid_minus = grid.copy()
        grid_minus[i, :] -= step_size
        divergence += (vector_field(*grid_plus)[..., i] - vector_field(*grid_minus)[..., i]) / (2 * step_size)
    
    return divergence
