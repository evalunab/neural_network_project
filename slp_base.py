"""
Base class for SLP Classifier and Regressor
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from activations import (
    relu,
    relu_derivative,
    tanh,
    tanh_derivative,
    logistic,
    logistic_derivative,
)


class BaseSLPEstimator(ABC):
    """
    Abstract base class for SLP Classifier and Regressor.

    Contains common functionality shared between both estimators.
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
        Initialize the SLP estimator.

        Parameters:
        -----------
        hidden_layer_size : int
            Number of neurons in the hidden layer
        activation : str
            Activation function ('logistic', 'tanh', 'relu'), default='logistic'
        learning_rate : float
            Learning rate for gradient descent
        max_iter : int
            Maximum number of iterations
        random_state : int or None
            Random seed for reproducibility
        tol : float
            Minimum improvement in loss per iteration to be considered progress.
        n_iter_no_change : int or None
            Number of consecutive iterations without improvement (> tol) before
            early stopping is triggered. Set to None to disable early stopping.
        """
        self.hidden_layer_size: int = hidden_layer_size
        self.activation: str = activation
        self.learning_rate: float = learning_rate
        self.max_iter: int = max_iter
        self.random_state: Optional[int] = random_state
        self.tol: float = tol
        self.n_iter_no_change: Optional[int] = n_iter_no_change

        # To be initialized in fit()
        self.W1_: Optional[NDArray[np.floating]] = None  # Weights: input -> hidden
        self.b1_: Optional[NDArray[np.floating]] = None  # Biases: hidden layer
        self.W2_: Optional[NDArray[np.floating]] = None  # Weights: hidden -> output
        self.b2_: Optional[NDArray[np.floating]] = None  # Biases: output layer
        self.loss_curve_: list[float] = []  # Loss at each iteration
        self.n_iter_: int = 0               # Actual number of iterations run

        # Dimensions set during fit
        self.n_features_in_: Optional[int] = None
        self.n_outputs_: Optional[int] = None

    def _get_activation_function(
        self,
    ) -> Tuple[Callable[[NDArray], NDArray], Callable[[NDArray], NDArray]]:
        """Return the activation function and its derivative."""
        if self.activation == "relu":
            return relu, relu_derivative
        elif self.activation == "tanh":
            return tanh, tanh_derivative
        elif self.activation == "logistic":
            return logistic, logistic_derivative
        else:
            raise ValueError(
                f"Unknown activation '{self.activation}'. "
                "Choose from: 'relu', 'tanh', 'logistic'."
            )

    def _initialize_weights(self) -> None:
        """
        Initialize weights using Xavier (Glorot) uniform initialization and
        biases to zero.

        Xavier init sets the range to sqrt(6 / (fan_in + fan_out)), which keeps
        activations in a reasonable range at the start and speeds convergence
        compared to naive small-random initialization.

        Requires self.n_features_in_ and self.n_outputs_ to be set beforehand.
        """
        rng = np.random.default_rng(self.random_state)

        # Hidden layer: Xavier initialization
        limit1 = np.sqrt(6.0 / (self.n_features_in_ + self.hidden_layer_size))
        self.W1_ = rng.uniform(-limit1, limit1,
                               size=(self.n_features_in_, self.hidden_layer_size))
        self.b1_ = np.zeros(self.hidden_layer_size)

        # Output layer: Xavier initialization
        limit2 = np.sqrt(6.0 / (self.hidden_layer_size + self.n_outputs_))
        self.W2_ = rng.uniform(-limit2, limit2,
                               size=(self.hidden_layer_size, self.n_outputs_))
        self.b2_ = np.zeros(self.n_outputs_)

    def _check_early_stopping(self, no_improve_count: int, loss: float, best_loss: float) -> Tuple[int, float, bool]:
        """
        Check whether early stopping should be triggered.

        Parameters:
        -----------
        no_improve_count : int
            Current number of consecutive iterations without sufficient improvement.
        loss : float
            Current iteration loss.
        best_loss : float
            Best loss seen so far.

        Returns:
        --------
        no_improve_count, best_loss, should_stop : updated state and stop flag
        """
        if self.n_iter_no_change is None:
            return 0, best_loss, False

        if loss < best_loss - self.tol:
            # Improvement found — reset counter and update best
            return 0, loss, False
        else:
            no_improve_count += 1
            if no_improve_count >= self.n_iter_no_change:
                return no_improve_count, best_loss, True
            return no_improve_count, best_loss, False

    @abstractmethod
    def _forward_propagation(self, X: NDArray[np.floating]) -> Tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
    ]:
        """Perform forward propagation. Must be implemented by subclasses."""
        pass

    @abstractmethod
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
        """Perform backpropagation. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _compute_loss(
        self, y_true: NDArray[np.floating], y_pred: NDArray[np.floating]
    ) -> float:
        """Compute loss. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def fit(self, X: NDArray[np.floating], y: NDArray) -> "BaseSLPEstimator":
        """Fit the model to training data. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def predict(self, X: NDArray[np.floating]) -> NDArray:
        """Make predictions. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def score(self, X: NDArray[np.floating], y: NDArray) -> float:
        """Score the model. Must be implemented by subclasses."""
        pass
