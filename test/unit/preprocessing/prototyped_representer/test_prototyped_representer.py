from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from fairlearn.preprocessing import PrototypedRepresenter


@parametrize_with_checks(
    [
        PrototypedRepresenter(max_iter=50),
    ],
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


def test_reconstruction():
    X = np.array([[10, 10], [20, 20]])
    y = np.array([0, 1])
    frl = PrototypedRepresenter(n_prototypes=2, lambda_z=0.0, lambda_y=0.0, random_state=42)
    frl.fit(X, y, sensitive_features=np.array([0, 1]))
    X_transformed = frl.transform(X)
    np.testing.assert_allclose(X_transformed, X, atol=1e-4)
    np.testing.assert_allclose(frl.prototypes_, X, atol=1e-4)


def test_statistical_parity():
    X = np.array([[10, 10], [20, 20], [30, 30], [40, 40]])
    y = np.array([0, 1, 0, 1])
    sensitive_features = np.array([0, 0, 1, 1])
    frl = PrototypedRepresenter(n_prototypes=4, lambda_x=0.0, lambda_y=0.0, random_state=42)
    frl.fit(X, y, sensitive_features=sensitive_features)

    M = frl._get_latent_mapping(X, frl.prototypes_, frl.alpha_)
    M_gk = np.array([np.mean(M[sensitive_features == group], axis=0) for group in frl._groups])
    # Compute the mean difference between prototype probabilities for each group
    group_combinations = np.triu_indices(n=len(frl._groups), k=1)
    fairness_error = np.mean(
        np.abs(M_gk[group_combinations[0], None] - M_gk[group_combinations[1], None])
    )
    np.testing.assert_allclose(fairness_error, 0.0, atol=1e-4)


def test_classification():
    X = np.array([[10, 10], [200, 200], [10, 10], [300, 300]])
    y = np.array([0, 1, 0, 1])
    sensitive_features = np.array([0, 1, 0, 1])
    frl = PrototypedRepresenter(n_prototypes=4, lambda_x=0.0, lambda_z=0.0, random_state=42)
    frl.fit(X, y, sensitive_features=sensitive_features)

    classification_error = frl.score(X, y)
    np.testing.assert_allclose(classification_error, 1.0, atol=1e-4)


@pytest.mark.parametrize(
    ["X", "prototypes", "alpha", "expected_M"],
    [
        (
            np.array([[10, 10], [20, 20]]),
            np.array([[10, 10], [20, 20]]),
            np.array([1, 1]),
            np.eye(2),
        ),
        (
            np.array([[10, 10], [20, 20]]),
            np.array([[10, 10], [20, 20]]),
            np.array([1, 10]),
            np.eye(2),
        ),
        (
            np.array([[20, 20], [10, 10]]),
            np.array([[10, 10], [20, 20]]),
            np.array([1, 10]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
        ),
        (
            np.array([[20, 20], [10, 10]]),
            np.array([[10, 10], [20, 20], [10, 10]]),
            np.array([1, 10]),
            np.array([[0.0, 1.0, 0.0], [0.5, 0.0, 0.5]]),
        ),
        (
            np.array([[20, 20], [10, 10]]),
            np.array([[0, 0], [0.0, 0.0], [20, 20]]),
            np.array([1, 1]),
            np.array([[0.0, 0.0, 1.0], [1 / 3, 1 / 3, 1 / 3]]),
        ),
        (
            np.array([[20, 15], [10, 10]]),
            np.array([[0.0, 5.0], [0.0, 5.0], [20, 15]]),
            np.array([1, 2]),
            np.array([[0.0, 0.0, 1.0], [1 / 3, 1 / 3, 1 / 3]]),
        ),
    ],
)
def test__get_latent_mapping(
    X: np.ndarray, prototypes: np.ndarray, alpha: np.ndarray, expected_M: np.ndarray
):
    M = PrototypedRepresenter._get_latent_mapping(X, prototypes, alpha)
    np.testing.assert_allclose(M, expected_M, atol=1e-4)


@pytest.mark.parametrize(
    ["y", "expected_y_transformed"],
    [
        (np.array([0, 1]), np.array([0, 1])),
        (np.array(["b", "a", "b"]), np.array([1, 0, 1])),
        (pd.Series(["yes", "yes"]), np.array([0, 0])),
    ],
)
def test__validate_X_y_maps_target_to_binary(y, expected_y_transformed: np.ndarray):
    X = np.eye(len(y))
    frl = PrototypedRepresenter()
    _, y_transformed = frl._validate_X_y(X, y)
    np.testing.assert_array_equal(y_transformed, expected_y_transformed)
    np.testing.assert_array_equal(
        frl._label_encoder.inverse_transform(y_transformed), np.asarray(y)
    )
