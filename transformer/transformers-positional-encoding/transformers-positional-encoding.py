import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    pos = np.arange(seq_length).reshape(-1, 1)
    i = np.arange(d_model)
    division_term = np.exp(2 * i * -np.log(10000) / d_model)
    pe = pos * division_term
    pe[:, 0::2] = np.sin(pe[:, 0::2])
    pe[:, 1::2] = np.cos(pe[:, 1::2])
    return pe