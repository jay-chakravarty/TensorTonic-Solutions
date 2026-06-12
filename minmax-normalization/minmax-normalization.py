import numpy as np

def minmax_scale(X, axis=0, eps=1e-12):
    """
    Scale X to [0,1]. If 2D and axis=0 (default), scale per column.
    Return np.ndarray (float).
    """
    # Write code here
    numerator = X - np.min(X, axis=axis, keepdims=True)
    denominator = np.max(X, axis=axis, keepdims=True) - np.min(X, axis=axis, keepdims=True)
    X_norm = numerator / np.maximum(denominator, eps)
    return X_norm