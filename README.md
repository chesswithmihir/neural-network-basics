# How to make a Neural Network
## By Mihir Mirchandani

### **The Philosophy: "Numpy is the CPU not GPU"**

To learn the *math*, you must use the CPU (Central Processing Unit) and `NumPy`. Libraries that use the GPU (like PyTorch or TensorFlow) handle the memory and math for you implicitly. You need to feel the pain of matrix multiplication dimensions clashing. Once you can do it in NumPy, switching to PyTorch on a Mac (`mps` device) will take 10 minutes.

-----

### **The Toolset**

Before starting Part A, you must adopt the two "Tricks" mentioned in your prompt.

#### **1. Einsum (Einstein Summation)**

Usually, matrix multiplication is written `np.matmul(A, B)`. This hides what is happening to the dimensions.
`einsum` forces you to write the indices explicitly.

  * **Concept:** You define the input shapes and the output shape using a string.
  * **Example:** Matrix Multiplication.
      * Matrix A is shape $(i, k)$. Matrix B is shape $(k, j)$. Output is $(i, j)$.
      * Code: `np.einsum('ik, kj -> ij', A, B)`

#### **2. Noam's Suffix Notation**

This is a naming convention to stop you from going crazy. You append the dimensions to the variable name.

  * `B`: Batch size (number of examples)
  * `T`: Time/Sequence length (number of words)
  * `C` or `D`: Channels/Dimension (size of the vector)

Instead of naming a variable `x`, you name it `x_BTD`. Now you know exactly what shape it is just by looking at the name.

-----

### **Part A: The Foundation (The "Core" Section)**

**Goal:** Build a 2-Layer MLP (Multi-Layer Perceptron) that can "learn" a simple pattern (like XOR).
**What is it?** An MLP is the simplest Neural Network. It takes numbers in, multiplies them by weights, adds a bias, runs them through a filter (ReLU), and spits an answer out.

**The Loop:**

1.  **Forward (FWD):** Calculate the prediction.
2.  **Loss:** Calculate how wrong the prediction is.
3.  **Backward (BWD):** Calculate the "gradient" (how much to nudge the weights to make the error smaller).

**The Daily Grind (Pen & Paper):**
Draw a circle for inputs, lines for weights, and a circle for outputs. Write the equation $y = Wx + b$.

**Your Coding Scaffold:**
Save this as `part_a_mlp.py`. You must fill in the `...`.

```python
import numpy as np

# --- The Tricks ---
# Use np.einsum for ALL matrix multiplications.

class TwoLayerMLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # 1. Initialize weights (W) and biases (b) with random small numbers
        # Shapes:
        # W1: (Input, Hidden) -> Suffix: W1_IH
        # b1: (Hidden)        -> Suffix: b1_H
        # W2: (Hidden, Output)-> Suffix: W2_HO
        # b2: (Output)        -> Suffix: b2_O
        self.params = {} 
        ...

    def relu(self, x):
        # Return x if x > 0, else 0
        ...

    def relu_derivative(self, x):
        # If x > 0, derivative is 1. If x <= 0, derivative is 0.
        # Used during Backward pass.
        ...

    def forward(self, X_BI):
        # X_BI means Input Data with shape (Batch_Size, Input_Dim)
        
        # 1. Linear Layer 1: Z1 = X @ W1 + b1 (Use einsum!)
        # 2. Activation: A1 = relu(Z1)
        # 3. Linear Layer 2: Z2 = A1 @ W2 + b2
        # 4. Store A1, Z1, X_BI for the backward pass
        ...
        return Z2_BO

    def backward(self, dL_dZ2_BO, learning_rate=1e-3):
        # dL_dZ2_BO is the "Gradient of Loss with respect to Output"
        # This is the "chain rule" in action.
        
        # 1. Calculate gradient for W2: dL_dW2
        # 2. Calculate gradient for b2: dL_db2
        # 3. Backprop error to hidden layer: dL_dA1
        # 4. Backprop through ReLU: dL_dZ1
        # 5. Calculate gradient for W1: dL_dW1
        # 6. Calculate gradient for b1: dL_db1
        
        # 7. Update parameters: W = W - learning_rate * grad
        ...

# --- Test Data (XOR Problem) ---
# Inputs: [0,0], [0,1], [1,0], [1,1]
X_BI = np.array([[0,0], [0,1], [1,0], [1,1]])
# Targets: [0], [1], [1], [0]
Y_BO = np.array([[0], [1], [1], [0]])

# Run training loop here...
```

-----

### **Part B: The Transformer (The "ML-Heavy" Section)**

**Goal:** Implement "Self-Attention" using Suffix Notation.
**What is it?** This is the math that makes ChatGPT work. It looks at a sentence and calculates how much each word "relates" to every other word.

**Concepts to learn:**

1.  **Q, K, V:** Query, Key, Value (Projections of the input).
2.  **Softmax:** Turns scores into probabilities (percentages).
3.  **Masking:** Ensuring the model can't "cheat" by looking at future words (for text generation).

**Your Coding Scaffold:**
Save this as `part_b_attention.py`.

```python
import numpy as np

def softmax(x):
    # Exponentiate and normalize
    ...

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
    
    # 2. Attention Scores
    # We want to multiply Q and K-transpose to get a (Time, Time) map.
    # Formula: Scores = (Q @ K.T) / sqrt(Dimension)
    # Hint: einsum('btd, bSd -> btS', Q, K) -> Note: S is also Time dimension
    
    # 3. Softmax
    # Apply softmax to the last dimension of Scores
    
    # 4. Aggregation
    # Multiply Weights (probs) by Values (V)
    # Output = Attn_Weights @ V
    
    return Output_BTD
```

-----

### **Part C: The System (Parallelism & Sharding)**

**Goal:** Understand how to split a matrix across two "fake" GPUs.
**What is it?** Modern models are too big for one GPU. We slice the weight matrices into pieces.

  * **Row Parallel:** Cut the matrix horizontally.
  * **Column Parallel:** Cut the matrix vertically.

**The Exercise:**
You will take the MLP from Part A, but instead of one `W1`, you will have `W1_part1` and `W1_part2`.

**Your Coding Scaffold:**
Save as `part_c_sharding.py`.

```python
import numpy as np

# Imagine we have 2 GPUs.
# We want to do: Y = X @ W

def column_parallel_forward(X, W_full):
    # 1. Split W_full into two columns: W_col1, W_col2
    
    # GPU 1 does: Y1 = X @ W_col1
    # GPU 2 does: Y2 = X @ W_col2
    
    # 2. Concatenate Y1 and Y2 to get the full output
    # Verify that this matches X @ W_full
    ...

def row_parallel_forward(X, W_full):
    # 1. Split W_full into two rows: W_row1, W_row2
    # 2. We also have to split X into X_col1, X_col2
    
    # GPU 1 does: Y1 = X_col1 @ W_row1
    # GPU 2 does: Y2 = X_col2 @ W_row2
    
    # 3. Sum (All-Reduce) Y1 and Y2 to get result
    # Verify matches X @ W_full
    ...
```

-----

### **Paper Exercises (Fast & Painful)**

Do not skip the "Pen + Paper" part. Every morning, write these down until you memorize them.

1.  **Broadcasting Rules:**
      * If shapes are `(3, 1)` and `(3, 5)`, what is the result?
2.  **FLOPs counting:**
      * How many floating point operations in a Matrix Multiply of $(M, K) @ (K, N)$?
      * *Answer:* $2 \cdot M \cdot N \cdot K$ (roughly). Derive why.
3.  **Memory:**
      * If you have a parameter matrix of size $1000 \times 1000$ in `float32` (4 bytes), how much RAM does it take?
      * *Math:* $10^6 \times 4$ bytes = 4MB.

### **How to Start**

1.  Install Python and NumPy.
2.  Implement `part_a_mlp.py` **Forward pass only**. Do not look up the answer. Struggle with the shapes. Use `print(x.shape)` every single line if you have to.
3.  Try to implement the Backward pass (Gradients). This is the hardest part for a beginner. You will likely fail the first time. That is the learning.