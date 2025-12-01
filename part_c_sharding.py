import numpy as np

# Imagine we have 2 GPUs.
# We want to do: Y = X @ W

def column_parallel_forward(X, W_full):
    """
    Simulate column parallel forward pass.
    
    X: Input matrix (Batch, Input_Dim)
    W_full: Full weight matrix (Input_Dim, Output_Dim)
    
    Returns: Output matrix (Batch, Output_Dim)
    """
    # 1. Split W_full into two columns: W_col1, W_col2
    # Split the weight matrix along the second dimension (columns)
    W_cols = np.hsplit(W_full, 2)  # Split into 2 equal parts
    W_col1 = W_cols[0]  # First half of columns
    W_col2 = W_cols[1]  # Second half of columns

    # 2. GPU 1 does: Y1 = X @ W_col1
    #    GPU 2 does: Y2 = X @ W_col2
    Y1 = np.dot(X, W_col1)
    Y2 = np.dot(X, W_col2)

    # 3. Concatenate Y1 and Y2 to get the full output
    # Verify that this matches X @ W_full
    Y_full = np.concatenate([Y1, Y2], axis=1)
    
    # Verify the result matches full matrix multiplication
    expected = np.dot(X, W_full)
    
    print("Column Parallel Forward:")
    print(f"X shape: {X.shape}")
    print(f"W_full shape: {W_full.shape}")
    print(f"W_col1 shape: {W_col1.shape}")
    print(f"W_col2 shape: {W_col2.shape}")
    print(f"Y_full shape: {Y_full.shape}")
    print(f"Expected shape: {expected.shape}")
    print(f"Results match: {np.allclose(Y_full, expected)}")
    
    return Y_full

def row_parallel_forward(X, W_full):
    """
    Simulate row parallel forward pass.
    
    X: Input matrix (Batch, Input_Dim)
    W_full: Full weight matrix (Input_Dim, Output_Dim)
    
    Returns: Output matrix (Batch, Output_Dim)
    """
    # 1. Split W_full into two rows: W_row1, W_row2
    #    Also split X into two parts: X_row1, X_row2
    # Split the weight matrix along the first dimension (rows)
    W_rows = np.vsplit(W_full, 2)  # Split into 2 equal parts
    W_row1 = W_rows[0]  # First half of rows
    W_row2 = W_rows[1]  # Second half of rows
    
    # Split X into two parts along the second dimension (columns)
    # This is a bit tricky - we need to split the input to match the weight split
    # For demonstration, let's assume we split X in the same way as W
    X_cols = np.hsplit(X, 2)  # Split input into 2 equal parts
    X_col1 = X_cols[0]  # First half of columns
    X_col2 = X_cols[1]  # Second half of columns
    
    # 2. GPU 1 does: Y1 = X_col1 @ W_row1
    #    GPU 2 does: Y2 = X_col2 @ W_row2
    Y1 = np.dot(X_col1, W_row1)
    Y2 = np.dot(X_col2, W_row2)

    # 3. Sum (All-Reduce) Y1 and Y2 to get result
    # Verify matches X @ W_full
    Y_full = Y1 + Y2
    
    # Verify the result matches full matrix multiplication
    expected = np.dot(X, W_full)
    
    print("\nRow Parallel Forward:")
    print(f"X shape: {X.shape}")
    print(f"W_full shape: {W_full.shape}")
    print(f"W_row1 shape: {W_row1.shape}")
    print(f"W_row2 shape: {W_row2.shape}")
    print(f"X_col1 shape: {X_col1.shape}")
    print(f"X_col2 shape: {X_col2.shape}")
    print(f"Y_full shape: {Y_full.shape}")
    print(f"Expected shape: {expected.shape}")
    print(f"Results match: {np.allclose(Y_full, expected)}")
    
    return Y_full

# Test the implementations
if __name__ == "__main__":
    # Create sample data
    # Input: 3 examples, 4 features
    X = np.random.randn(3, 4)
    
    # Weight matrix: 4 inputs, 6 outputs
    W_full = np.random.randn(4, 6)
    
    print("Testing Column Parallel Forward:")
    Y_col = column_parallel_forward(X, W_full)
    
    print("\nTesting Row Parallel Forward:")
    Y_row = row_parallel_forward(X, W_full)
    
    print("\nVerification:")
    expected = np.dot(X, W_full)
    print(f"Full matrix multiplication result shape: {expected.shape}")
    print(f"Column parallel matches: {np.allclose(Y_col, expected)}")
    print(f"Row parallel matches: {np.allclose(Y_row, expected)}")