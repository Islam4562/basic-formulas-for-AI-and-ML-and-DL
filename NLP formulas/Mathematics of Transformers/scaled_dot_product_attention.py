import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Implementation of the scaled dot-product attention mechanism with optional masking.
    
    Args:
    Q: (batch_size, num_heads, seq_len, d_k) - Query tensor
    K: (batch_size, num_heads, seq_len, d_k) - Key tensor
    V: (batch_size, num_heads, seq_len, d_v) - Value tensor
    mask: (batch_size, 1, seq_len, seq_len) - Masking tensor (optional), used to prevent attention to future tokens

    Returns:
    output: (batch_size, num_heads, seq_len, d_v) - Attention output
    attention_weights: (batch_size, num_heads, seq_len, seq_len) - Attention weight matrix
    """
    d_k = Q.shape[-1]  # Dimension of key vectors

    # Compute scaled dot-product scores: QK^T / sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    # Apply masking if provided (useful for preventing future token attention in autoregressive models)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Apply softmax along the attention dimension to obtain attention weights
    attention_weights = F.softmax(scores, dim=-1)

    # Compute the final output by multiplying attention weights with the value matrix
    output = torch.matmul(attention_weights, V)

    return output, attention_weights  # Return both the attention output and weights

# ===================== Testing =====================
batch_size = 2   # Number of sequences in the batch
num_heads = 4    # Number of attention heads
seq_len = 5      # Length of the input sequence
d_k = 8         # Dimensionality of query/key vectors
d_v = 8         # Dimensionality of value vectors

# Generate random Q, K, V tensors
Q = torch.randn(batch_size, num_heads, seq_len, d_k)
K = torch.randn(batch_size, num_heads, seq_len, d_k)
V = torch.randn(batch_size, num_heads, seq_len, d_v)

# Create a mask to simulate causal masking (used in GPT-like models)
mask = torch.tril(torch.ones(seq_len, seq_len))  # Lower triangular matrix (allows attention only to past and current tokens)
mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch_size and num_heads dimensions
mask = mask.expand(batch_size, num_heads, seq_len, seq_len)  # Expand to match input size

# Call the attention function
output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

print("Output shape:", output.shape)  # Expected: (batch_size, num_heads, seq_len, d_v)
print("Attention Weights shape:", attention_weights.shape)  # Expected: (batch_size, num_heads, seq_len, seq_len)
