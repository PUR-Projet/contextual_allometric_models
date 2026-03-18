from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import curve_fit

class Method(ABC):

    @property
    @abstractmethod
    def shortname(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X, y):
        pass

    @abstractmethod
    def nb_parameters(self) -> int:
        pass


class OracleChave(Method):

    shortname = "oc"
    name = "Oracle Chave"

    def fit(self, X, y):
        pass

    def predict(self, X, y):
        # X = DBH,H,W,C
        return 0.0673 * (X[:, 2] * X[:, 0] ** 2 * X[:, 1]) ** 0.976


class Chave(Method):
    shortname = "c"
    name = "Chave method"

    def __init__(self):
        self.alpha = 1.0
        self.beta = 1.0
        self.epsilon = 0.0

    def z(self, X):
        return X[:, 2] * X[:, 1] * X[:, 0] ** 2

    @staticmethod
    def log_chave(z, alpha, beta):
        return alpha + beta * np.log(z)

    @staticmethod
    def chave_with_epsilon(z, alpha, beta, epsilon):
        return np.exp(alpha + epsilon) * (z**beta)

    def fit(self, X, y):
        z = self.z(X)
        ly = np.log(y)
        p0 = (1, 1)
        popt, pcov = curve_fit(Chave.log_chave, z, ly, p0, method="lm")
        self.alpha = popt[0]
        self.beta = popt[1]
        self.epsilon = self.compute_epsilon(X, y)
        return self

    def compute_epsilon(self, X, y):
        z = self.z(X)
        ly = np.log(y)
        ly_hat = Chave.log_chave(z, self.alpha, self.beta)
        residuals = ly_hat - ly
        N = len(y)
        p = 2  # alpha & beta
        sigma = np.sqrt(np.sum(residuals**2) / (N - p))
        return (sigma**2) / 2

    def predict(self, X, y=None):
        z = self.z(X)
        return Chave.chave_with_epsilon(z, self.alpha, self.beta, self.epsilon)

    def nb_parameters(self) -> int:
        return 2
