import numpy as np

class Perceptron:
    def __init__(self, eta=0.01, n_iter=10, init_bias=None):
        self.eta = eta               
        self.n_iter = n_iter         
        self.init_bias = init_bias       

    def weighted_sum(self, X):
        return np.dot(X, self.w_) + self.bias_

    def activation_function(self, weighted_sum):
        return 1 if weighted_sum >= 0 else -1

    def predict(self, X):
        return np.array([self.activation_function(self.weighted_sum(xi)) for xi in X])

    def fit(self, X, y):
        self.w_ = np.random.normal(loc=0.0, scale=0.01, size=X.shape[1])
        
        if self.init_bias is not None:
            self.bias_ = self.init_bias
        else:
            self.bias_ = 0.0  

        self.errors_ = []  

        for _ in range(self.n_iter):
            errors = 0
            print("HIIII I AM IN PRECEPTON")
            for xi, target in zip(X, y):
                
                y_pred = self.activation_function(self.weighted_sum(xi))
                update = self.eta * (target - y_pred)

                self.w_ += update * xi
                self.bias_ += update
                errors += int(update != 0.0)  

            self.errors_.append(errors)
        return self
