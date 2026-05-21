"""
SimpleSLPClassifier - Single Layer Perceptron for Classification
"""

from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from activations import softmax
from slp_base import BaseSLPEstimator


class SimpleSLPClassifier(BaseSLPEstimator):
    """
    Simple Single Layer Perceptron Classifier with one hidden layer.

    Compatible interface with sklearn.neural_network.MLPClassifier.

    Architecture:
        Input -> [W1, b1] -> Hidden (activation) -> [W2, b2] -> Output (softmax)

    Supports binary and multi-class classification via one-hot encoding
    and softmax output. Uses cross-entropy loss and the Adam optimizer.
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
        Initialize the SLP classifier.

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

        # Classifier-specific attributes (set during fit)
        self.classes_: Optional[NDArray[np.int_]] = None
        self.n_outputs_: Optional[int] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _one_hot_encode(self, y: NDArray[np.int_]) -> NDArray[np.floating]:
        """Convert integer class labels to a one-hot matrix."""
        n_samples = len(y)
        n_classes = len(self.classes_)
        label_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        indices = np.array([label_to_idx[label] for label in y])
        Y_ohe = np.zeros((n_samples, n_classes))
        Y_ohe[np.arange(n_samples), indices] = 1.0
        return Y_ohe

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

        Hidden layer  : z1 = X @ W1 + b1 ;  a1 = activation(z1)
        Output layer  : z2 = a1 @ W2 + b2 ; y_pred = softmax(z2)
        """
        activation_fn, _ = self._get_activation_function()
        z1 = X @ self.W1_ + self.b1_
        a1 = activation_fn(z1)
        z2 = a1 @ self.W2_ + self.b2_
        y_pred = softmax(z2)
        return z1, a1, z2, y_pred

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
        Backpropagation for cross-entropy + softmax output.

        The gradient of cross-entropy w.r.t. z2 simplifies to (y_pred - y_true) / N,
        a well-known result that avoids computing the full softmax Jacobian.
        """
        _, activation_deriv = self._get_activation_function()
        n_samples = X.shape[0]

        # Output layer gradient
        delta2 = (y_pred - y) / n_samples
        dW2 = a1.T @ delta2
        db2 = np.sum(delta2, axis=0)

        # Hidden layer gradient
        delta1 = (delta2 @ self.W2_.T) * activation_deriv(z1)
        dW1 = X.T @ delta1
        db1 = np.sum(delta1, axis=0)

        return dW1, db1, dW2, db2

    def _compute_loss(
        self, y_true: NDArray[np.floating], y_pred: NDArray[np.floating]
    ) -> float:
        """
        Mean cross-entropy loss.

        L = -1/N * sum(y_true * log(y_pred))
        """
        y_pred_clipped = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self, X: NDArray[np.floating], y: NDArray[np.int_]
    ) -> "SimpleSLPClassifier":
        """
        Fit the SLP classifier to training data.

        Uses full-batch gradient descent with the Adam optimizer.
        Training stops either after `max_iter` iterations or when
        early stopping is triggered (if `n_iter_no_change` is set).

        Parameters:
        -----------
        X : array, shape (n_samples, n_features)
        y : array, shape (n_samples,)  integer class labels

        Returns:
        --------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # Identify classes and encode targets
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.n_outputs_ = len(self.classes_)
        Y_ohe = self._one_hot_encode(y)

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

            # Forward pass + loss
            z1, a1, z2, y_pred = self._forward_propagation(X)
            loss = self._compute_loss(Y_ohe, y_pred)
            self.loss_curve_.append(loss)

            # Early stopping check
            no_improve_count, best_loss, should_stop = self._check_early_stopping(
                no_improve_count, loss, best_loss
            )
            if should_stop:
                break

            # Backward pass
            dW1, db1, dW2, db2 = self._backward_propagation(X, Y_ohe, z1, a1, z2, y_pred)

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

    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Predict class probabilities for X.

        Returns:
        --------
        proba : array, shape (n_samples, n_classes)
        """
        X = np.asarray(X, dtype=float)
        _, _, _, y_pred = self._forward_propagation(X)
        return y_pred

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.int_]:
        """
        Predict class labels for X.

        Returns:
        --------
        y_pred : array, shape (n_samples,)  original class labels
        """
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    def score(self, X: NDArray[np.floating], y: NDArray[np.int_]) -> float:
        """
        Return mean accuracy on the given test data.

        Returns:
        --------
        accuracy : float in [0, 1]
        """
        return float(np.mean(self.predict(X) == np.asarray(y)))
