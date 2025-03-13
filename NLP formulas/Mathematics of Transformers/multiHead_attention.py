import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Multi-Head Attention Layer
        
        :param embed_dim: Dimension of input embeddings
        :param num_heads: Number of attention heads
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # Dimension per attention head

        # Linear layers for query, key, and value projections
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        # Final linear projection after multi-head attention
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Computes Scaled Dot-Product Attention.
        
        :param Q: Query tensor (batch, num_heads, seq_len, head_dim)
        :param K: Key tensor (batch, num_heads, seq_len, head_dim)
        :param V: Value tensor (batch, num_heads, seq_len, head_dim)
        :param mask: Optional mask (batch, 1, seq_len, seq_len) 
        
        :return: Attention output (batch, num_heads, seq_len, head_dim)
        """
        d_k = Q.shape[-1]  # Key dimension
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))  # Apply masking if provided

        attention_weights = F.softmax(scores, dim=-1)  # Apply softmax to obtain attention weights
        output = torch.matmul(attention_weights, V)  # Multiply by values

        return output

    def forward(self, Q, K, V, mask=None):
        """
        Forward pass of Multi-Head Attention.
        
        :param Q: Input queries (batch, seq_len, embed_dim)
        :param K: Input keys (batch, seq_len, embed_dim)
        :param V: Input values (batch, seq_len, embed_dim)
        :param mask: Optional mask (batch, 1, seq_len, seq_len)
        
        :return: Output tensor (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = Q.shape

        # 1. Apply linear projections
        Q = self.W_q(Q)  # (batch, seq_len, embed_dim)
        K = self.W_k(K)  # (batch, seq_len, embed_dim)
        V = self.W_v(V)  # (batch, seq_len, embed_dim)

        # 2. Reshape into multiple heads: (batch, seq_len, embed_dim) -> (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Compute attention
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # 4. Concatenate heads back: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, embed_dim)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # 5. Apply final linear transformation
        output = self.W_o(attention_output)

        return output

# ===================== Testing =====================
batch_size = 2
seq_len = 5
embed_dim = 32  # Embedding dimension (must be divisible by num_heads)
num_heads = 4

# Generate random input data
Q = torch.randn(batch_size, seq_len, embed_dim)
K = torch.randn(batch_size, seq_len, embed_dim)
V = torch.randn(batch_size, seq_len, embed_dim)

# Generate mask (if needed)
mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
mask = mask.expand(batch_size, num_heads, seq_len, seq_len)  # Expand to match batch and head dimensions

# Create Multi-Head Attention layer
mha = MultiHeadAttention(embed_dim, num_heads)

# Forward pass
output = mha(Q, K, V, mask)

print("Output shape:", output.shape)  # Expected: (batch_size, seq_len, embed_dim)
