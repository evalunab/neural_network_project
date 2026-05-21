"""
SimpleSLPRegressor - Single Layer Perceptron for Regression
"""

from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from slp_base import BaseSLPEstimator


class SimpleSLPRegressor(BaseSLPEstimator):
    """
    Simple Single Layer Perceptron Regressor with one hidden layer.

    Compatible interface with sklearn.neural_network.MLPRegressor.

    Architecture:
        Input -> [W1, b1] -> Hidden (activation) -> [W2, b2] -> Output (identity)

    Uses MSE loss and the Adam optimizer. Targets are normalised internally
    (zero mean, unit variance) so that the learning rate is scale-independent.
    Optionally stops early when the training loss stops improving.
    """

    def __init__(
        self,
        hidden_layer_size: int = 100,
        activation: str = "logistic",
        learning_rate: float = 0.001,
        max_iter: int = 200,
        random_state: Optional[int] = None,
        tol: float = 1e-4,
        n_iter_no_change: Optional[int] = 10,
    ) -> None:
        """
        Initialize the SLP regressor.

        Parameters:
        -----------
        hidden_layer_size : int
            Number of neurons in the hidden layer
        activation : str
            Activation function for the hidden layer
            ('logistic', 'tanh', 'relu'), default='logistic'
        learning_rate : float
            Learning rate for the Adam optimizer
        max_iter : int
            Maximum number of gradient descent iterations
        random_state : int or None
            Random seed for weight initialization reproducibility
        tol : float
            Minimum loss improvement per iteration to count as progress.
        n_iter_no_change : int or None
            Consecutive iterations without improvement before early stopping.
            Set to None to disable.
        """
        super().__init__(
            hidden_layer_size, activation, learning_rate, max_iter,
            random_state, tol, n_iter_no_change,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _forward_propagation(
        self, X: NDArray[np.floating]
    ) -> Tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
    ]:
        """
        Forward propagation through the network.

        Hidden layer : z1 = X @ W1 + b1 ;  a1 = activation(z1)
        Output layer : z2 = a1 @ W2 + b2 ; y_pred = z2  (linear / identity)
        """
        activation_fn, _ = self._get_activation_function()
        z1 = X @ self.W1_ + self.b1_
        a1 = activation_fn(z1)
        z2 = a1 @ self.W2_ + self.b2_
        return z1, a1, z2, z2   # y_pred = z2 (identity output)

    def _backward_propagation(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        z1: NDArray[np.floating],
        a1: NDArray[np.floating],
        z2: NDArray[np.floating],
        y_pred: NDArray[np.floating],
    ) -> Tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
    ]:
        """
        Backpropagation for MSE loss with linear output.

        dL/dz2 = (y_pred - y) / N
        """
        _, activation_deriv = self._get_activation_function()
        n_samples = X.shape[0]

        delta2 = (y_pred - y) / n_samples
        dW2 = a1.T @ delta2
        db2 = np.sum(delta2, axis=0)

        delta1 = (delta2 @ self.W2_.T) * activation_deriv(z1)
        dW1 = X.T @ delta1
        db1 = np.sum(delta1, axis=0)

        return dW1, db1, dW2, db2

    def _compute_loss(
        self, y_true: NDArray[np.floating], y_pred: NDArray[np.floating]
    ) -> float:
        """
        Mean squared error loss.

        L = 1/N * sum((y_true - y_pred)^2)
        """
        return float(np.mean((y_true - y_pred) ** 2))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self, X: NDArray[np.floating], y: NDArray[np.floating]
    ) -> "SimpleSLPRegressor":
        """
        Fit the SLP regressor to training data.

        Uses full-batch gradient descent with the Adam optimizer.
        Targets are normalised internally so that the fixed learning rate
        works across different target scales. Training stops either after
        `max_iter` iterations or when early stopping is triggered.

        Parameters:
        -----------
        X : array, shape (n_samples, n_features)
        y : array, shape (n_samples,) or (n_samples, n_outputs)

        Returns:
        --------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Ensure y is 2-D
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            self._single_output = True
        else:
            self._single_output = False

        self.n_features_in_ = X.shape[1]
        self.n_outputs_ = y.shape[1]

        # Normalise targets to zero mean and unit variance for stable training
        self._y_mean = y.mean(axis=0)
        self._y_std = y.std(axis=0)
        self._y_std[self._y_std == 0] = 1.0   # avoid division by zero
        y_scaled = (y - self._y_mean) / self._y_std

        # Weight initialisation
        self._initialize_weights()
        self.loss_curve_ = []
        self.n_iter_ = 0

        # Adam optimizer state
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        mW1 = np.zeros_like(self.W1_); vW1 = np.zeros_like(self.W1_)
        mb1 = np.zeros_like(self.b1_); vb1 = np.zeros_like(self.b1_)
        mW2 = np.zeros_like(self.W2_); vW2 = np.zeros_like(self.W2_)
        mb2 = np.zeros_like(self.b2_); vb2 = np.zeros_like(self.b2_)

        # Early stopping state
        best_loss = np.inf
        no_improve_count = 0

        for t in range(1, self.max_iter + 1):
            self.n_iter_ = t

            # Forward pass + loss (on normalised targets)
            z1, a1, z2, y_pred = self._forward_propagation(X)
            loss = self._compute_loss(y_scaled, y_pred)
            self.loss_curve_.append(loss)

            # Early stopping check
            no_improve_count, best_loss, should_stop = self._check_early_stopping(
                no_improve_count, loss, best_loss
            )
            if should_stop:
                break

            # Backward pass
            dW1, db1, dW2, db2 = self._backward_propagation(X, y_scaled, z1, a1, z2, y_pred)

            # Adam moment updates
            mW1 = beta1 * mW1 + (1 - beta1) * dW1
            vW1 = beta2 * vW1 + (1 - beta2) * dW1 ** 2
            mb1 = beta1 * mb1 + (1 - beta1) * db1
            vb1 = beta2 * vb1 + (1 - beta2) * db1 ** 2
            mW2 = beta1 * mW2 + (1 - beta1) * dW2
            vW2 = beta2 * vW2 + (1 - beta2) * dW2 ** 2
            mb2 = beta1 * mb2 + (1 - beta1) * db2
            vb2 = beta2 * vb2 + (1 - beta2) * db2 ** 2

            # Bias-corrected parameter updates
            mW1h = mW1 / (1 - beta1 ** t); vW1h = vW1 / (1 - beta2 ** t)
            mb1h = mb1 / (1 - beta1 ** t); vb1h = vb1 / (1 - beta2 ** t)
            mW2h = mW2 / (1 - beta1 ** t); vW2h = vW2 / (1 - beta2 ** t)
            mb2h = mb2 / (1 - beta1 ** t); vb2h = vb2 / (1 - beta2 ** t)

            self.W1_ -= self.learning_rate * mW1h / (np.sqrt(vW1h) + eps)
            self.b1_ -= self.learning_rate * mb1h / (np.sqrt(vb1h) + eps)
            self.W2_ -= self.learning_rate * mW2h / (np.sqrt(vW2h) + eps)
            self.b2_ -= self.learning_rate * mb2h / (np.sqrt(vb2h) + eps)

        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Predict using the trained model.

        Predictions are automatically de-normalised back to the original target scale.

        Parameters:
        -----------
        X : array, shape (n_samples, n_features)

        Returns:
        --------
        y_pred : array, shape (n_samples,) or (n_samples, n_outputs)
        """
        X = np.asarray(X, dtype=float)
        _, _, _, y_pred = self._forward_propagation(X)
        y_pred = y_pred * self._y_std + self._y_mean   # de-normalise

        if self._single_output:
            return y_pred.ravel()
        return y_pred

    def score(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> float:
        """
        Return the R² (coefficient of determination) score.

        R² = 1 - SS_res / SS_tot

        Parameters:
        -----------
        X : array, shape (n_samples, n_features)
        y : array, shape (n_samples,) or (n_samples, n_outputs)

        Returns:
        --------
        r2 : float
        """
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0

        return float(1.0 - ss_res / ss_tot)
