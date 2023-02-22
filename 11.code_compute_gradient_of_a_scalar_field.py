import numpy as np

def gradient(scalar_field, step_size=0.01):
    """
    Compute the gradient of a scalar field.
    """
    # Get the shape of the scalar field
    shape = np.array(scalar_field.shape)
    
    # Create a grid of points
    grid = np.array(np.meshgrid(*[np.arange(i) for i in shape], indexing='ij')).astype(float)
    
    # Reshape the scalar field to a vector
    scalar_field = scalar_field.flatten()
    
    # Compute the gradient of the scalar field by finite differences
    gradient = np.zeros((len(scalar_field), len(grid)))
    for i in range(len(grid)):
        grid_plus = grid.copy()
        grid_plus[i, :] += step_size
        grid_minus = grid.copy()
        grid_minus[i, :] -= step_size
        gradient[:, i] = (scalar_field(*grid_plus) - scalar_field(*grid_minus)) / (2 * step_size)
    
    # Reshape the gradient back to the shape of the scalar field
    gradient = gradient.reshape(*shape, -1)
    
    return gradient
