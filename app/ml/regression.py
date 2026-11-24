"""Regressao linear multipla implementada pelo grupo, treinada por gradiente descendente.

Modelo:   y_hat = X @ w + b            (5 pesos, um por lag, e um vies)
Perda:    MSE = (1/n) * sum((y_hat - y)^2)
Gradiente:
    dMSE/dw = (2/n) * X.T @ (y_hat - y)
    dMSE/db = (2/n) * sum(y_hat - y)
Atualizacao (a cada epoca, com todas as linhas = gradiente descendente em lote):
    w <- w - alpha * dMSE/dw
    b <- b - alpha * dMSE/db

Por que o min-max e necessario aqui: o passo alpha so e estavel se for menor que
2/L, onde L e a maior curvatura da perda. Com as features em [0, 1], L <= 2*(5+1) = 12,
entao qualquer alpha < 1/6 converge para qualquer base. Sem normalizar (valores ate ~700),
L fica na casa de 10^5 e o mesmo alpha = 0.1 diverge em poucas epocas.

Segue a interface do scikit-learn (fit/predict), para encaixar no Pipeline com o
MinMaxScaler e na validacao cruzada.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class GradientDescentRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, learning_rate: float = 0.1, max_epochs: int = 100_000, tol: float = 1e-6):
        self.learning_rate = learning_rate  # alpha: tamanho do passo
        self.max_epochs = max_epochs  # limite de seguranca
        self.tol = tol  # para quando o gradiente fica menor que isso

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, n_features = X.shape

        w = np.zeros(n_features)  # comeca com todos os pesos em zero
        b = 0.0
        losses = []

        for epoch in range(self.max_epochs):
            residual = X @ w + b - y  # erro de cada linha (previsto - real)
            with np.errstate(over="ignore", invalid="ignore"):
                loss = float(np.mean(residual**2))  # MSE desta epoca
            # Se a perda explodiu, o passo e grande demais para a escala dos dados.
            if not np.isfinite(loss) or loss > 1e100:
                raise ValueError(
                    "O gradiente descendente divergiu: reduza learning_rate "
                    "ou normalize as features."
                )
            losses.append(loss)

            grad_w = (2 / n) * (X.T @ residual)
            grad_b = (2 / n) * residual.sum()

            # Convergiu: o gradiente e praticamente zero, estamos no minimo.
            if np.sqrt(grad_w @ grad_w + grad_b**2) < self.tol:
                break

            w -= self.learning_rate * grad_w
            b -= self.learning_rate * grad_b

        self.coef_ = w
        self.intercept_ = b
        self.n_epochs_ = epoch + 1
        self.converged_ = epoch + 1 < self.max_epochs
        self.loss_history_ = losses
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_
