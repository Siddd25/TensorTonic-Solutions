import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    n, d = X.shape

    W = np.zeros(d)
    b = 0.0

    for i in range(steps):

        # Forward propagation
        z = X @ W + b
        p = _sigmoid(z)

        # Calculate loss
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)

        loss = -np.mean(
            y * np.log(p) + (1-y) * np.log(1-p)
        )

        # Calculate gradients
        del_W = X.T @ (p-y) / n
        del_b = np.sum(p-y) / n

        # Update parameters
        W -= lr * del_W
        b -= lr * del_b

    return W, b