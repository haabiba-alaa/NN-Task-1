import numpy as np

class Adaline:
    def __init__(self, eta=0.01, n_iter=10, init_weights=None, init_bias=None, init_threshold=0.05):
        self.eta = eta               # Learning rate
        self.n_iter = n_iter         # Number of iterations
        self.init_weights = init_weights  # Optional initial weights
        self.init_bias = init_bias        # Optional initial bias
        self.init_threshold = init_threshold

    def weighted_sum(self, X):
        # Compute the weighted sum plus bias
        return np.dot(X, self.w_) + self.bias_

    def activation_function(self, weighted_sum):
        # ADALINE uses the weighted sum directly as the output during training
        return weighted_sum

    def predict(self, X):
        # Apply the threshold to the output for final predictions
        # Return 1 if output >= 0, else -1
        return np.where(self.weighted_sum(X) >= 0.0, 1, -1)

    def fit(self, X, y):
        # Initialize weights and bias
        if self.init_weights is not None:
            self.w_ = np.array(self.init_weights)
        else:
            self.w_ = np.random.normal(loc=0.0, scale=0.01, size=X.shape[1])
        
        if self.init_bias is not None:
            self.bias_ = self.init_bias
        else:
            self.bias_ = 0.0  # Default bias term

        self.errors_ = []  # Track MSE per iteration
       
        for _ in range(self.n_iter):
            # Calculate the net input (weighted sum)
            net_input = self.weighted_sum(X)
            
            # Calculate the continuous error without applying the activation function
            errors = y - net_input
            
            # Update weights and bias based on the continuous error
            self.w_ += self.eta * X.T.dot(errors)
            self.bias_ += self.eta * errors.sum()
            
            # Calculate Mean Squared Error (MSE) and append to error list
            mse = np.mean(errors ** 2)
            self.errors_.append(mse)

            # Stop if the MSE is below the threshold
            if mse < self.init_threshold:
                break
        
        return self
