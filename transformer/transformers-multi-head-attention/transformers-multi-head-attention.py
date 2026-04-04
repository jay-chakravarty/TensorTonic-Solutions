import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
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