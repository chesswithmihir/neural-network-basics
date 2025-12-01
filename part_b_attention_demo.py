import numpy as np

def softmax(x):
    # Exponentiate and normalize
    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def self_attention(X_BTD, W_q_DD, W_k_DD, W_v_DD):
    """
    X_BTD: Input batch (Batch, Time/Sequence, Dimension)
    W_q, W_k, W_v: Weight matrices for projections
    """

    # 1. Projections
    # Q = X @ W_q
    # K = X @ W_k
    # V = X @ W_v
    # Use einsum! Shapes will be (Batch, Time, Dim)
    
    Q_BTD = np.einsum('btd, dd -> btd', X_BTD, W_q_DD)
    K_BTD = np.einsum('btd, dd -> btd', X_BTD, W_k_DD)
    V_BTD = np.einsum('btd, dd -> btd', X_BTD, W_v_DD)

    # 2. Attention Scores
    # We want to multiply Q and K-transpose to get a (Time, Time) map.
    # Formula: Scores = (Q @ K.T) / sqrt(Dimension)
    # Hint: einsum('btd, bSd -> btS', Q, K) -> Note: S is also Time dimension
    
    # Compute attention scores
    # Shape: (B, T, T) - each position attends to all positions
    scores_BTT = np.einsum('btd, bTd -> btT', Q_BTD, K_BTD) / np.sqrt(X_BTD.shape[-1])

    # 3. Softmax
    # Apply softmax to the last dimension of Scores
    weights_BTT = softmax(scores_BTT)

    # 4. Aggregation
    # Multiply Weights (probs) by Values (V)
    # Output = Attn_Weights @ V
    Output_BTD = np.einsum('btT, bTd -> btd', weights_BTT, V_BTD)
    
    return Output_BTD

# Test the implementation with a simple example
if __name__ == "__main__":
    # Create a simple 1x3x2 input (1 batch, 3 time steps, 2 dimensions)
    X_BTD = np.array([[[1.0, 2.0],
                       [3.0, 4.0],
                       [5.0, 6.0]]])
    
    # Initialize weight matrices (2x2)
    W_q_DD = np.array([[1.0, 0.0],
                       [0.0, 1.0]])
    W_k_DD = np.array([[1.0, 0.0],
                       [0.0, 1.0]])
    W_v_DD = np.array([[1.0, 0.0],
                       [0.0, 1.0]])
    
    print("Input X_BTD:")
    print(X_BTD)
    print("\nWeight matrices:")
    print("W_q_DD:")
    print(W_q_DD)
    print("W_k_DD:")
    print(W_k_DD)
    print("W_v_DD:")
    print(W_v_DD)
    
    # Run attention
    output_BTD = self_attention(X_BTD, W_q_DD, W_k_DD, W_v_DD)
    
    print("\nOutput shape:", output_BTD.shape)
    print("Output:")
    print(output_BTD)
    
    # Show the attention weights for better understanding
    Q_BTD = np.einsum('btd, dd -> btd', X_BTD, W_q_DD)
    K_BTD = np.einsum('btd, dd -> btd', X_BTD, W_k_DD)
    scores_BTT = np.einsum('btd, bTd -> btT', Q_BTD, K_BTD) / np.sqrt(X_BTD.shape[-1])
    weights_BTT = softmax(scores_BTT)
    
    print("\nAttention weights (how much each position attends to others):")
    print(weights_BTT)