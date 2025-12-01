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
        # Initialize weights with He initialization for ReLU
        self.params['W1_IH'] = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.params['b1_H'] = np.zeros((hidden_dim,))
        self.params['W2_HO'] = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.params['b2_O'] = np.zeros((output_dim,))

    def relu(self, x):
        # Return x if x > 0, else 0
        return np.maximum(0, x)

    def relu_derivative(self, x):
        # If x > 0, derivative is 1. If x <= 0, derivative is 0.
        # Used during Backward pass.
        return (x > 0).astype(float)

    def forward(self, X_BI):
        # X_BI means Input Data with shape (Batch_Size, Input_Dim)

        # 1. Linear Layer 1: Z1 = X @ W1 + b1 (Use einsum!)
        # 2. Activation: A1 = relu(Z1)
        # 3. Linear Layer 2: Z2 = A1 @ W2 + b2
        # 4. Store A1, Z1, X_BI for the backward pass
        self.cache = {}
        self.cache['X_BI'] = X_BI

        # Linear layer 1 with einsum
        Z1_IH = np.einsum('bi, ih -> bh', X_BI, self.params['W1_IH'])
        Z1_IH += self.params['b1_H']

        # Activation
        A1_BH = self.relu(Z1_IH)
        self.cache['A1_BH'] = A1_BH
        self.cache['Z1_IH'] = Z1_IH

        # Linear layer 2 with einsum
        Z2_BO = np.einsum('bh, ho -> bo', A1_BH, self.params['W2_HO'])
        Z2_BO += self.params['b2_O']

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
        batch_size = self.cache['X_BI'].shape[0]

        # Gradient w.r.t. W2 and b2
        dL_dW2_HO = np.einsum('bh, bo -> ho', self.cache['A1_BH'], dL_dZ2_BO) / batch_size
        dL_db2_O = np.sum(dL_dZ2_BO, axis=0) / batch_size

        # Backprop to hidden layer
        dL_dA1_BH = np.einsum('bo, ho -> bh', dL_dZ2_BO, self.params['W2_HO'])

        # Backprop through ReLU
        dL_dZ1_IH = dL_dA1_BH * self.relu_derivative(self.cache['Z1_IH'])

        # Gradient w.r.t. W1 and b1
        dL_dW1_IH = np.einsum('bi, bh -> ih', self.cache['X_BI'], dL_dZ1_IH) / batch_size
        dL_db1_H = np.sum(dL_dZ1_IH, axis=0) / batch_size

        # Update parameters
        self.params['W2_HO'] -= learning_rate * dL_dW2_HO
        self.params['b2_O'] -= learning_rate * dL_db2_O
        self.params['W1_IH'] -= learning_rate * dL_dW1_IH
        self.params['b1_H'] -= learning_rate * dL_db1_H

    def mse_loss(self, y_pred_BO, y_true_BO):
        # Mean Squared Error loss
        return np.mean((y_pred_BO - y_true_BO) ** 2)

    def mse_loss_derivative(self, y_pred_BO, y_true_BO):
        # Derivative of MSE loss
        # We will divide by batch size later in backward pass
        return 2 * (y_pred_BO - y_true_BO)

# --- Test Data (XOR Problem) ---
# Inputs: [0,0], [0,1], [1,0], [1,1]
X_BI = np.array([[0,0], [0,1], [1,0], [1,1]])
# Targets: [0], [1], [1], [0]
Y_BO = np.array([[0], [1], [1], [0]])

# Create and train the MLP
mlp = TwoLayerMLP(input_dim=2, hidden_dim=4, output_dim=1)

# Training loop
for epoch in range(5000):
    # Forward pass
    y_pred_BO = mlp.forward(X_BI)

    # Calculate loss
    loss = mlp.mse_loss(y_pred_BO, Y_BO)

    # Calculate gradient of loss
    dL_dZ2_BO = mlp.mse_loss_derivative(y_pred_BO, Y_BO)

    # Backward pass
    mlp.backward(dL_dZ2_BO, learning_rate=1e-2)

    # Print loss every 500 epochs
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# Final predictions
final_predictions = mlp.forward(X_BI)
print("\nFinal predictions:")
for i, (input_val, target, prediction) in enumerate(zip(X_BI, Y_BO, final_predictions)):
    print(f"Input: {input_val}, Target: {target}, Prediction: {prediction[0]:.4f}")