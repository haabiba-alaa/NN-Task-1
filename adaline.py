import numpy as np

class Adaline:
    def __init__(self, eta=0.01, n_iter=10, init_bias=None, init_threshold=0.05):
        self.eta = eta               
        self.n_iter = n_iter         
        self.init_bias = init_bias        
        self.init_threshold = init_threshold

    def weighted_sum(self, X):
        return np.dot(X, self.w_) + self.bias_

    def predict(self, X):
        return np.where(self.weighted_sum(X) >= 0.0, 1, -1)

    def fit(self, X, y):
        #random init weights
        self.w_ = np.random.normal(loc=0.0, scale=0.01, size=X.shape[1])
        
        if self.init_bias is not None:
            self.bias_ = self.init_bias
        else:
            self.bias_ = 0.0 

        self.errors_ = []  # Track MSE
       
        for _ in range(self.n_iter):
            print("HIII I AM IN ADALINE")
            net_input = self.weighted_sum(X)
            
            errors = y - net_input
            
            self.w_ += self.eta * X.T.dot(errors)
            self.bias_ += self.eta * errors.sum()
            
            mse = np.mean(errors ** 2)
            self.errors_.append(mse)

            if mse < self.init_threshold:
               break
        
        return self