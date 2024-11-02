import numpy as np

class Perceptron:
    def __init__(self, eta=0.01, n_iter=10, init_weights=None, init_bias=None):
        self.eta = eta               # Learning rate
        self.n_iter = n_iter         # Number of iterations
        self.init_weights = init_weights  # Optional initial weights
        self.init_bias = init_bias        # Optional initial bias

    def weighted_sum(self, X):
        # Compute the weighted sum plus bias
        return np.dot(X, self.w_) + self.bias_

    def activation_function(self, weighted_sum):
        return 1 if weighted_sum >= 0 else -1

    def predict(self, X):
        # Apply activation function to each sample's weighted sum
        return np.array([self.activation_function(self.weighted_sum(xi)) for xi in X])

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

        self.errors_ = []  # Track errors per iteration

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                #print("HIIII")
                # Calculate prediction and update weights and bias
                y_pred = self.activation_function(self.weighted_sum(xi))
                update = self.eta * (target - y_pred)

                # Update weights and bias
                self.w_ += update * xi
                self.bias_ += update
                errors += int(update != 0.0)  # Increment error if there was an update

            self.errors_.append(errors)  # Record the number of errors for this iteration
        return self
