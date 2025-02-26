from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize
from scipy.spatial.distance import cdist
from scipy.special import softmax
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    TransformerMixin,
    check_is_fitted,
)
from sklearn.calibration import LabelEncoder
from sklearn.dummy import check_random_state
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.utils.multiclass import type_of_target

from fairlearn.utils._fixes import validate_data
from fairlearn.utils._input_validation import _validate_and_reformat_input

LOGGER = logging.getLogger(__name__)


class PrototypedRepresenter(ClassifierMixin, TransformerMixin, BaseEstimator):
    r"""
    A transformer and classifier that learns a latent representation of the input data to
    obfuscate the sensitive features while preserving the classification and reconstruction
    performance.

    The model minimizes a loss function that consists of three terms: the reconstruction error,
    the classification error, and an approximation of the demographic parity difference.

    Read more in the :ref:`User Guide <preprocessing>`.

    Parameters
    ----------
    n_prototypes : int, default=2
        Number of prototypes to use in the latent representation.

    lambda_x : float, default=1.0
        Weight of the reconstruction error term in the objective function.

    lambda_y : float, default=1.0
        Weight of the classification error term in the objective function.

    lambda_z : float, default=1.0
        Weight of the fairness error term in the objective function.

    random_state : int, np.random.RandomState, or None, default=None
        Seed or random number generator for reproducibility.

    tol : float, default=1e-6
        Convergence tolerance for the optimization algorithm.

    max_iter : int, default=1000
        Maximum number of iterations for the optimization algorithm.

    Attributes
    ----------
    n_prototypes : int
        Number of prototypes to use in the latent representation.

    lambda_x : float
        Weight of the reconstruction error term in the objective function.

    lambda_y : float
        Weight of the classification error term in the objective function.

    lambda_z : float
        Weight of the fairness error term in the objective function.

    random_state : int, np.random.RandomState, or None
        Seed or random number generator for reproducibility.

    tol : float
        Tolerance for the optimization algorithm.

    max_iter : int
        Maximum number of iterations for the optimization algorithm.

    coef_ : np.ndarray
        Coefficients of the learned model.

    n_iter_ : int
        Number of iterations run by the optimization algorithm.

    n_features_in_ : int
        Number of features in the input data.

    classes_ : np.ndarray
        Unique classes in the target variable.

    Notes
    -----
    The PrototypedRepresenter implements the algorithms intoduced in Zemel et al.
    :footcite:`pmlr-v28-zemel13`.

    If no sensitive features are provided during fitting, the model falls back to a Logistic
    Regression classifier.

    References
    ----------
    .. footbibliography::

    Examples
    --------
    >>> import numpy as np
    >>> from fairlearn.preprocessing import PrototypedRepresenter
    >>> X = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    >>> y = np.array([0, 1, 0, 1])
    >>> sensitive_features = np.array([0, 0, 1, 1])
    >>> frl = PrototypedRepresenter(n_prototypes=2, random_state=42)
    >>> frl.fit(X, y, sensitive_features=sensitive_features)
    >>> X_transformed = frl.transform(X)
    >>> y_pred = frl.predict(X)
    """

    n_prototypes: int
    lambda_x: float
    lambda_y: float
    lambda_z: float
    random_state: int | np.random.RandomState | None
    tol: float
    max_iter: int
    coef_: np.ndarray
    n_iter_: int
    n_features_in_: int
    classes_: np.ndarray
    _label_encoder: LabelEncoder
    _groups: pd.Series
    # The following attributes are set during fitting and can be None depending on whether
    # sensitive features were provided
    _prototypes_: np.ndarray | None
    _alpha: np.ndarray | None
    _prototype_dim: int | None
    _latent_mapping_size: int | None
    _prototype_predictions_size: int | None
    _prototype_vectors_size: int | None
    _optimizer_size: int | None
    _fall_back_classifier: LogisticRegression | None

    def __init__(
        self,
        n_prototypes: int = 2,
        lambda_x: float = 1.0,
        lambda_y: float = 1.0,
        lambda_z: float = 1.0,
        random_state: int | np.random.RandomState | None = None,
        tol: float = 1e-6,
        max_iter: int = 1000,
    ) -> None:
        self.n_prototypes = n_prototypes
        self.lambda_z = lambda_z
        self.lambda_x = lambda_x
        self.lambda_y = lambda_y
        self.random_state = random_state
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y, *, sensitive_features=None):
        """
        Fit the prototyped representer to the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        y : array-like of shape (n_samples,)
            The target values.

        sensitive_features : array-like or None, default=None
            Sensitive features to be considered whose groups will be used to promote demographic
            parity. If None, the model will fall back to a Logistic Regression classifier.

        Returns
        -------
        self : PrototypedRepresenter
            Returns the fitted instance.
        """
        X, y = self._validate_X_y(X, y)

        _, _, sensitive_features, _ = _validate_and_reformat_input(
            X,
            y,
            sensitive_features=sensitive_features,
            expect_y=True,
            expect_sensitive_features=False,
            enforce_binary_labels=True,
        )

        self.n_features_in_ = X.shape[1]
        random_state = check_random_state(self.random_state)

        if sensitive_features is None:
            LOGGER.warning("No sensitive features provided. Fitting a Logistic Regression.")

            return self._optimize_without_sensitive_features(X, y, random_state)

        return self._optimize_with_sensitive_features(X, y, sensitive_features, random_state)

    def _optimize_with_sensitive_features(
        self, X, y, sensitive_features: pd.Series, random_state: np.random.RandomState
    ):
        """
        Minimize the loss given the sensitive features.

        This method sets up and executes the optimization algorithm by:
        - Initializing the optimization variables: the prototype vectors and their predictions are
          randomly drawn, and the dimension weights are initialized to one.
        - Defining the constraints: prototype predictions are constrained to [0,1], and
          dimension weights must be non-negative.
        - Executing the "L-BFGS-B" optimizer, extracting and reshaping the resulting optimal
          values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.
        sensitive_features : pd.Series
            The sensitive features for each sample.
        random_state : np.random.RandomState
            The random state for reproducibility. Used for initializing the optimization.

        Returns
        -------
        self : PrototypedRepresenter
            Returns self.

        Raises
        ------
        RuntimeError
            If the loss minimization fails.
        """
        self._groups = sensitive_features.unique()

        # Dimension of each v_k prototype vector
        self._prototype_dim = X.shape[1]
        # Dimension of M, the latent representation stochastic mapping from X to Z
        self._latent_mapping_size = len(self._groups) * self.n_prototypes
        # Dimension of the prototype predictions vector w
        self._prototype_predictions_size = self.n_prototypes
        # Total size of the prototype vectors
        self._prototype_vectors_size = self.n_prototypes * self._prototype_dim

        self._optimizer_size = (
            self.n_prototypes * self._prototype_dim  # V
            + self._prototype_predictions_size  # w
            + self._prototype_dim  # alpha: the weight of each dimension in the distance computation
        )

        # Initialize the prototype vectors v_k
        V0 = random_state.rand(self.n_prototypes, self._prototype_dim)

        # Initialize the prototype predictions w_k
        w0 = random_state.rand(self.n_prototypes)

        # Initialize the dimension weights alpha
        alpha0 = np.ones(self._prototype_dim)

        x0 = np.concatenate([V0.flatten(), w0, alpha0])

        bounds = (
            [(None, None)] * self._prototype_vectors_size  # The prototype vectors are unbounded
            + [(0, 1)]
            * self._prototype_predictions_size  # The prototype predictions are in [0, 1]
            + [(0, None)] * self._prototype_dim  # The dimension weights are non-negative
        )

        try:
            result: OptimizeResult = minimize(
                self._objective,
                x0=x0,
                bounds=bounds,
                args=(X, y, sensitive_features),
                method="L-BFGS-B",
                tol=self.tol,
                options={"maxiter": self.max_iter},
            )
        except Exception as optimization_error:
            raise RuntimeError("The loss minimization failed.") from optimization_error

        self.coef_ = result.x[self._prototype_vectors_size : -self._prototype_dim]
        self._prototypes_ = result.x[: self._prototype_vectors_size].reshape(
            (self.n_prototypes, self._prototype_dim)
        )
        self._alpha_ = result.x[-self._prototype_dim :]
        self.n_iter_ = result.nit

        self._fall_back_classifier = None

        return self

    def _objective(self, x: np.ndarray, X, y, sensitive_features) -> float:
        """
        Compute the objective function for the optimization problem at the given point :code:`x`.

        This method extracts the current prototype vectors, the prototype predictions, and the
        dimension weights, and calculates the current loss, which is the weighted sum of the
        reconstruction error, classification error, and fairness error.
        Parameters
        ----------
        x : np.ndarray
            The optimization variable containing prototype vectors, weights, and dimension weights.
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.
        sensitive_features : pd.Series
            The sensitive features for each sample.
        Returns
        -------
        float
            The computed objective value.
        """

        assert x.shape == (self._optimizer_size,)
        # Compute the reconstruction error
        V = x[: self._prototype_vectors_size].reshape((self.n_prototypes, self._prototype_dim))
        alpha = x[-self._prototype_dim :]
        M = self._get_latent_mapping(X, V, dimension_weights=alpha)
        X_hat = M @ V
        reconstruction_error = np.mean(np.sum((X - X_hat) ** 2, axis=1))

        # Compute the fairness error
        # Compute the mean prototype probabilities for each group
        M_gk = np.array(
            [np.mean(M[sensitive_features == group], axis=0) for group in self._groups]
        )
        # Compute the mean difference between mean prototype probabilities for each group
        group_combinations = np.triu_indices(n=len(self._groups), k=1)
        fairness_error = np.mean(
            np.abs(M_gk[group_combinations[0], None] - M_gk[group_combinations[1], None])
        )

        # Compute the classification error
        w = x[self._prototype_vectors_size : -self._prototype_dim]
        y_hat = M @ w
        classification_error = log_loss(y, y_hat)

        return (
            self.lambda_x * reconstruction_error
            + self.lambda_y * classification_error
            + self.lambda_z * fairness_error
        )

    def _optimize_without_sensitive_features(self, X, y, random_state: np.random.RandomState):
        """
        Optimize the model without considering sensitive features.

        This method trains a logistic regression model on the provided features and labels,
        and stores the resulting coefficients and other relevant attributes.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            The input samples.

        y : array-like of shape (n_samples,)
            The target values.

        random_state : np.random.RandomState
            The random state to use for reproducibility.

        Returns
        -------
        self : PrototypedRepresenter
            Returns the instance itself.
        """
        self._groups = pd.Series()

        fallback_classifier = LogisticRegression(
            tol=self.tol, max_iter=self.max_iter, random_state=random_state
        )

        self._fall_back_classifier = fallback_classifier.fit(X, y)

        self.coef_ = self._fall_back_classifier.coef_
        self._prototypes_ = None
        self._alpha_ = None
        self.n_iter_ = self._fall_back_classifier.n_iter_

        return self

    def transform(self, X) -> np.ndarray:
        """
        Transform the input data X using the learned prototyped representation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to transform.

        Returns
        -------
        np.ndarray
            The transformed data.

        Notes
        -----
        This method checks if the model is fitted, validates the input data,
        and then applies the learned prototyped representation.
        If a fallback classifier is set, it returns the input data as is.
        Otherwise, it computes the latent mapping and returns the transformed data.
        """
        check_is_fitted(self)

        X = validate_data(self, X, reset=False)

        if self._fall_back_classifier is not None:
            return X

        M = self._get_latent_mapping(X, self._prototypes_, dimension_weights=self._alpha_)
        return M @ self._prototypes_

    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities for the input samples X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
            The class probabilities of the input samples. The first column
            represents the probability of the negative class, and the second
            column represents the probability of the positive class.

        Raises
        ------
        NotFittedError
            If the estimator is not fitted yet.
        """
        check_is_fitted(self)

        X = validate_data(self, X, reset=False)

        if self._fall_back_classifier is not None:
            return self._fall_back_classifier.predict_proba(X)

        M = self._get_latent_mapping(X, self._prototypes_, dimension_weights=self.alpha_)
        positive_proba = M @ self.coef_
        return np.c_[1 - positive_proba, positive_proba]

    def predict(self, X) -> np.ndarray:
        """
        Predict the labels for the given input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to predict.

        Returns
        -------
        np.ndarray
            The predicted labels for the input data.
        """
        check_is_fitted(self)

        X = validate_data(self, X, reset=False)

        binary_predictions = (self.predict_proba(X)[:, 1] > 0.5).astype(int)

        return self._label_encoder.inverse_transform(binary_predictions)

    @property
    def prototypes_(self) -> np.ndarray:
        check_is_fitted(self)

        if self._prototypes_ is None:
            raise AttributeError(
                "No sensitive features provided when fitting. No prototypes were learned."
            )

        return self._prototypes_

    @property
    def alpha_(self) -> np.ndarray:
        check_is_fitted(self)

        if self._alpha_ is None:
            raise AttributeError(
                "No sensitive features provided when fitting. No distance was learned."
            )

        return self._alpha_

    @staticmethod
    def _get_latent_mapping(
        X, prototypes: np.ndarray, dimension_weights: np.ndarray
    ) -> np.ndarray:
        """
        Compute the latent mapping of the input data X to the given prototypes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to be mapped.
        prototypes : np.ndarray of shape (n_prototypes, n_features)
            The prototype vectors to which the input data will be mapped.
        dimension_weights : np.ndarray of shape (n_features,)
            The weights for each dimension used in the distance calculation.

        Returns
        -------
        np.ndarray of shape (n_samples, n_prototypes)
            The latent mapping of the input data to the prototypes, where each
            element represents the softmax-transformed negative distance between
            a sample and a prototype.
        """
        distances = cdist(X, prototypes, metric="euclidean", w=dimension_weights)
        M = softmax(-distances, axis=1)
        return M

    def _validate_X_y(self, X, y) -> tuple[np.ndarray, np.ndarray]:
        """
        Validate and preprocess the input features and target labels.

        Parameters
        ----------
        X : array-like
            The input features.
        y : array-like
            The target labels.

        Returns
        -------
        tuple of np.ndarray
            The validated and preprocessed input features and target labels.

        Raises
        ------
        ValueError
            If the target labels are not binary.
        """
        X, y = validate_data(self, X, y=y, allow_nd=True, ensure_2d=False, ensure_all_finite=True)

        y_type = type_of_target(y, input_name="y")
        if y_type != "binary":
            raise ValueError(
                f"Unknown label type: {y_type}. Only binary classification is supported."
            )
        self.classes_ = np.unique(y)
        self._label_encoder = LabelEncoder().fit(y)
        y = self._label_encoder.transform(y)

        return X, y

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags

    def _more_tags(self):
        return {"binary_only": True}
