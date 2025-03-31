import numpy as np

def bce(y_true, y_pred, epsilon=1e-12):
    """
    Compute Binary Cross-Entropy (BCE) between ground truth and predicted scores.

    Args:
    y_true (list or np.array): Ground truth probabilities (values between 0 and 1).
    y_pred (list or np.array): Predicted probabilities (values between 0 and 1).
    epsilon (float): Small value to avoid log(0).

    Returns:
    float: BCE loss.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Clip predictions to prevent log(0) issues
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Compute BCE
    bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    return bce

def mse(y_true, y_pred):
    """
    Compute Mean Squared Error (MSE) loss for adjacency matrices.

    Args:
    A_true (np.array): Ground truth adjacency matrix.
    A_pred (np.array): Predicted adjacency matrix.

    Returns:
    float: MSE loss.
    """
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)