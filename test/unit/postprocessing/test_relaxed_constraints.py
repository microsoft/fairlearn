from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from fairlearn.postprocessing._relaxed_constraints import (
    maximize_objective_with_tolerance,
)
from fairlearn.postprocessing._threshold_optimizer import (
    OBJECTIVES_FOR_SIMPLE_CONSTRAINTS,
    SIMPLE_CONSTRAINTS,
    ThresholdOptimizer,
)

from .conftest import ExamplePredictor


@pytest.mark.parametrize(
    ["x", "ys", "weights", "tol", "expected_indices", "expected_maximum"],
    [
        (
            np.linspace(0, 1, 10),
            [np.linspace(0, 1, 10) for _ in range(3)],
            [1 / 3, 1 / 3, 1 / 3],
            0.0,
            [9, 9, 9],
            1.0,
        ),
        (
            np.array([0, 0.5, 1]),
            [np.array([0, 1.0, 1.0]), np.array([0.0, 1.0, 1.0])],
            [0.5, 0.5],
            0.5,
            [2, 2],
            1.0,
        ),
        (
            np.linspace(0, 1, 3),
            [np.array([0, 2, 0]), np.array([0, 1, 2])],
            [0.5, 0.5],
            0.6,
            [1, 2],
            2.0,
        ),
        (
            np.array([0, 0.5, 1]),
            [np.array([1, 0.2, 0.0]), np.array([0.1, 0.2, 0.6]), np.array([0.0, 0.1, 0.5])],
            [0.1, 0.8, 0.1],
            0.6,
            [1, 2, 2],
            0.55,
        ),
        (
            np.linspace(0, 1, 6),
            [np.array([1, 0, 0, 0, 1, 0]), np.array([1, 0, 0, 0, 0, 1])],
            [0.5, 0.5],
            0.2,
            [0, 0],
            1.0,
        ),
    ],
)
def test_maximize_objective_with_tolerance_returns_correct_values(
    x: NDArray,
    ys: list[NDArray],
    weights: NDArray,
    tol: float,
    expected_indices: list[int],
    expected_maximum: float,
) -> None:
    dataframes = [pd.DataFrame({"x": x, "y": y}) for y in ys]
    optimal_indices, maximum = maximize_objective_with_tolerance(
        dataframes=dataframes, weights=weights, tol=tol
    )

    assert expected_indices == optimal_indices
    assert expected_maximum == maximum


@pytest.mark.parametrize("objective", OBJECTIVES_FOR_SIMPLE_CONSTRAINTS)
@pytest.mark.parametrize("constraints", SIMPLE_CONSTRAINTS)
def test_threshold_optimization_with_tolerance_increases_objective_and_respects_constraints(
    constraints: str, objective: str
) -> None:
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([1, 0, 1, 1, 0])
    sensitive_features = np.array(["a", "b", "a", "b", "a"])
    tol = 0.2

    threshold_optimizer = partial(
        ThresholdOptimizer,
        estimator=ExamplePredictor(scores),
        constraints=constraints,
        objective=objective,
        flip=True,
        predict_method="predict",
        grid_size=50,
    )

    strict_predictor = threshold_optimizer(tol=None)
    relaxed_predictor = threshold_optimizer(tol=tol)

    strict_predictor.fit(X, y, sensitive_features=sensitive_features)
    relaxed_predictor.fit(X, y, sensitive_features=sensitive_features)

    assert relaxed_predictor._y_best >= strict_predictor._y_best
    assert relaxed_predictor._between_groups <= tol
