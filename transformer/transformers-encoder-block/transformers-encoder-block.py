import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    # Your code here
    mu = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(variance + eps) + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    # Your code here
    h = num_heads
    B, N, d_model = Q.shape
    d_k = d_model // h
    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v
    Q_heads = Q_proj.reshape(B, N, h, d_k).transpose(0, 2, 1, 3)
    K_heads = K_proj.reshape(B, N, h, d_k).transpose(0, 2, 1, 3)
    V_heads = V_proj.reshape(B, N, h, d_k).transpose(0, 2, 1, 3)
    scores = softmax(Q_heads @ K_heads.transpose(0, 1, 3, 2) / np.sqrt(d_k))
    attention = scores @ V_heads
    attention = attention.transpose(0, 2, 1, 3).reshape(B, N, d_model)
    return attention @ W_o

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    # Your code here
    hidden = np.dot(x, W1) + b1
    relu_out = np.maximum(0, hidden)
    return np.dot(relu_out, W2) + b2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    # Your code here
    mha = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    x_prime = layer_norm(x + mha, gamma1, beta1)
    ffn = feed_forward(x_prime, W1, b1, W2, b2)
    output = layer_norm(x_prime + ffn, gamma2, beta2)
    return output