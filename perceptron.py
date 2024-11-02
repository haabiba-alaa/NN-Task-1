import numpy as np
class Perceptron:
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def weighted_sum(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def signum(self, v):
        return np.where(v >= 0, 1, -1)

    def predict(self, X):
        return self.signum(self.weighted_sum(X))

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                v = self.weighted_sum(xi)
                y_pred = self.signum(v)
                update = self.eta * (target - y_pred)
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
