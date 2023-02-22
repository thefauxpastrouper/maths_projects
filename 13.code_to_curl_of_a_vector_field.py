import numpy as np

def curl(vector_field, step_size=0.01):
    """
    Compute the curl of a vector field.
    """
    # Get the shape of the vector field
    shape = np.array(vector_field.shape[:-1])
    
    # Create a grid of points
    grid = np.array(np.meshgrid(*[np.arange(i) for i in shape], indexing='ij')).astype(float)
    
    # Compute the curl of the vector field by finite differences
    curl = np.zeros(shape + (3,))
    for i, j in [(0, 1), (1, 2), (2, 0)]:
        grid_plus = grid.copy()
        grid_plus[j, :] += step_size
        grid_minus = grid.copy()
        grid_minus[j, :] -= step_size
        curl[..., i] = (vector_field(*grid_plus)[..., j] - vector_field(*grid_minus)[..., j]) / (2 * step_size)
    
    # Compute the curl in the opposite direction
    for i, j in [(1, 2), (2, 0), (0, 1)]:
        grid_plus = grid.copy()
        grid_plus[i, :] += step_size
        grid_minus = grid.copy()
        grid_minus[i, :] -= step_size
        curl[..., i] -= (vector_field(*grid_plus)[..., j] - vector_field(*grid_minus)[..., j]) / (2 * step_size)
    
    return curl
