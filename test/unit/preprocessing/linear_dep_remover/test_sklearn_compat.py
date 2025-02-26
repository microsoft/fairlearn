# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from fairlearn.preprocessing import CorrelationRemover


@parametrize_with_checks(
    [
        CorrelationRemover(sensitive_feature_ids=[]),
        CorrelationRemover(sensitive_feature_ids=[0]),
    ],
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


@pytest.mark.parametrize(
    ["sensitive_feature_ids", "X", "expected_result"],
    [
        (
            [0],
            np.array(
                [
                    [0, 0, 1, 1],
                    [1, 1, 2, 2],
                    [0.1, 0.2, 1.2, 1.1],
                ]
            ).T,
            np.array([[1.5, 0.6], [1.5, 0.7], [1.5, 0.7], [1.5, 0.6]]),
        ),
        (
            ("a",),
            pd.DataFrame(
                np.array(
                    [
                        [0, 0, 1, 1],
                        [1, 1, 2, 2],
                        [0.1, 0.2, 1.2, 1.1],
                    ]
                ).T,
                columns=["a", "b", "c"],
            ),
            np.array([[1.5, 0.6], [1.5, 0.7], [1.5, 0.7], [1.5, 0.6]]),
        ),
        (
            ("a",),
            pl.DataFrame(
                np.array(
                    [
                        [0, 0, 1, 1],
                        [1, 1, 2, 2],
                        [0.1, 0.2, 1.2, 1.1],
                    ]
                ).T,
                schema=["a", "b", "c"],
            ),
            np.array([[1.5, 0.6], [1.5, 0.7], [1.5, 0.7], [1.5, 0.6]]),
        ),
    ],
)
def test_linear_dependence(sensitive_feature_ids, X, expected_result) -> None:
    X_tfm = CorrelationRemover(sensitive_feature_ids=sensitive_feature_ids).fit_transform(X)
    np.testing.assert_array_almost_equal(expected_result, X_tfm)


@pytest.mark.parametrize(
    ["alpha", "expected_result"],
    [
        (0.0, np.array([[2, 3], [5, 6], [8, 9]])),
        (0.5, np.array([[3.5, 4.5], [5, 6], [6.5, 7.5]])),
        (1.0, np.array([[5, 6], [5, 6], [5, 6]])),
    ],
)
def test_correlation_remover_handles_alpha(alpha, expected_result) -> None:
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    X_tfm = CorrelationRemover(sensitive_feature_ids=[0], alpha=alpha).fit_transform(X)
    np.testing.assert_array_almost_equal(expected_result, X_tfm)


@pytest.mark.parametrize(
    "X",
    [
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        pd.DataFrame({"a": [1, 4, 7], "b": [2, 5, 8], "c": [3, 6, 9]}),
        pl.DataFrame({"a": [1, 4, 7], "b": [2, 5, 8], "c": [3, 6, 9]}),
    ],
)
def test_correlation_remover_handles_default_sensitive_features(X) -> None:
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    X_tfm = CorrelationRemover().fit_transform(X)
    np.testing.assert_array_equal(np.array(X), X_tfm)


@pytest.mark.parametrize(
    ["sensitive_feature_ids", "X", "expectation"],
    [
        (None, np.array([]), does_not_raise()),
        ([], np.array([[1, 2], [3, 4]]), does_not_raise()),
        ([0], np.array([[1, 2], [3, 4]]), does_not_raise()),
        ((1, 0), np.array([[1, 2]]), does_not_raise()),
        (["a", "b"], pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]}), does_not_raise()),
        (["a", "b"], pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]}), does_not_raise()),
        (
            [-1, 0, 3, 4],
            np.array([[1, 2], [3, 4]]),
            pytest.raises(
                ValueError, match=re.escape("Columns [-1, 3, 4] not found in the input data.")
            ),
        ),
        (
            ["a", "b", "c", "d"],
            pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
            pytest.raises(
                ValueError, match=re.escape("Columns ['c', 'd'] not found in the input data.")
            ),
        ),
        (
            ["a", "b", "c", "d"],
            pl.DataFrame({"a": [1, 2], "b": [3, 4]}),
            pytest.raises(
                ValueError, match=re.escape("Columns ['c', 'd'] not found in the input data.")
            ),
        ),
    ],
)
def test__check_sensitive_features_in_X(sensitive_feature_ids, X, expectation) -> None:
    remover = CorrelationRemover(sensitive_feature_ids=sensitive_feature_ids)

    with expectation:
        remover._check_sensitive_features_in_X(X)
