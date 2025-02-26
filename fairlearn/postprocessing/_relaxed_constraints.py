from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np
import pandas as pd


def maximize_objective_with_tolerance(
    dataframes: Iterable[pd.DataFrame],
    weights: Iterable[float],
    tol: float,
    x_col: str = "x",
    y_col: str = "y",
) -> tuple[list[int], float]:
    """
    Finds the indices that maximize the sum of weighted `y` values across multiple DataFrames
    under the constraint that the range of the associated `x` values is at most equal to `tol`.

    Parameters
    -----------
    dataframes:  Iterable[pd.DataFrame]
        An Iterable of DataFrames. The DataFrames must have:
        - A common `x` column (sorted in ascending order).
        - A `y` column.
    weights: Iterable[float]:
        An Iterable of weights for each DataFrame.
    tol: float
        The maximum allowed tolerance.

    Returns
    -------
    tuple[list[int], float]:
        - A list of selected indices (one per DataFrame).
        - The maximum sum of weighted `y` values under the constraint.

    Algorithm
    ---------
    The algorithm uses a sliding window over sorted x values to enforce the tolerance constraint.
    For each DataFrame, a deque (double-ended queue) maintains indices in descending order of y
    values within the window at each iteration, ensuring the front always holds the current maximum.
    As the window advances, expired indices are removed, and new indices are added while preserving
    the descending order. The optimal indices for the current window are selected from the deque
    fronts, updating the solution if the weighted sum exceeds the previous maximum or, in cases of
    equality, if the x-range of the selected indices is smaller.


    """
    # Extract `x` values (all DataFrames must share the same `x` values)
    x_values = dataframes[0][x_col].values
    n = len(x_values)

    # Extract weighted `y` columns
    y_columns = [df[y_col].values * weight for df, weight in zip(dataframes, weights)]
    m = len(y_columns)

    # Initialize a deque for each DataFrame to store indices in descending order of y values
    deques = [deque() for _ in range(m)]

    # Variables to track the best solution at each iteration
    best_indices: list[int] = []
    max_objective = float("-inf")
    min_violation = float("inf")

    # Sliding window start pointer
    start = 0

    # Traverse with the end pointer
    for end in range(n):
        # Add the current `end` index to the deques, maintaining descending order of y values
        for k in range(m):
            while deques[k] and (y_columns[k][deques[k][-1]] <= y_columns[k][end]):
                deques[k].pop()
            deques[k].append(end)

        # Adjust the window to ensure the tolerance constraint is satisfied
        while x_values[end] - x_values[start] > tol:
            for k in range(m):
                if deques[k] and (deques[k][0] == start):
                    deques[k].popleft()

            start += 1

        # Calculate the maximum weighted sum of `y` values in the current window
        # Pick the first element from each deque since the indices are stored in descending order
        # of y_values
        current_indices = [deques[k][0] for k in range(m)]
        current_objective = sum(y_columns[k][current_indices[k]] for k in range(m))

        # Calculate the violation of the equality constraint in the current window
        current_violation = np.ptp(x_values[current_indices])

        # Update the best solution if the current objective is larger, or if the current objective
        # is equal and the current violation is smaller
        if current_objective > max_objective or (
            current_objective == max_objective and current_violation <= min_violation
        ):
            max_objective = current_objective
            best_indices = current_indices
            min_violation = current_violation

    return best_indices, max_objective
