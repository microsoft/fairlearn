# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.
from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import pandas as pd

from fairlearn.utils._input_validation import _validate_and_reformat_input

from .moment import _ALL, _LABEL, ClassificationMoment

_MESSAGE_BAD_COSTS = (
    "costs needs to be a dictionary with keys "
    "'fp' and 'fn' containing non-negative values, which are not both zero"
)


class ErrorRate(ClassificationMoment):
    r"""Misclassification error as a moment.

    A classifier :math:`h(X)` has the misclassification error equal to

    .. math::
      P[h(X) \ne Y]

    It is also possible to specify costs for false positives and false
    negatives. The error then evaluates to

    .. math::
      c_{FP} P[h(X)=1, Y=0] + c_{FN} P[h(X)=0, Y=1]

    where :math:`c_{FP}` and :math:`c_{FN}` are the costs of false positive
    and false negative errors respectively. The standard misclassification
    error corresponds to :math:`c_{FP}=c_{FN}=1.0`.

    Parameters
    ----------
    costs : dict
        The dictionary with keys :code:`'fp'` and :code:`'fn'` containing the
        costs of false positives and false negatives. If none are provided
        costs of 1.0 are assumed.
    """

    def __init__(self, *, costs: dict[Literal["fp", "fn"], float] | None = None):
        """Initialize the costs."""
        super(ErrorRate, self).__init__()
        if costs is None:
            self.fp_cost = 1.0
            self.fn_cost = 1.0
        elif (
            isinstance(costs, dict)
            and costs.keys() == {"fp", "fn"}
            and costs["fp"] >= 0.0
            and costs["fn"] >= 0.0
            and costs["fp"] + costs["fn"] > 0.0
        ):
            self.fp_cost = costs["fp"]
            self.fn_cost = costs["fn"]
        else:
            raise ValueError(_MESSAGE_BAD_COSTS)

    def load_data(self, X, y, *, sensitive_features, control_features=None) -> None:
        """Load the specified data into the object."""
        _, y_train, sf_train, _ = _validate_and_reformat_input(
            X,
            y,
            enforce_binary_labels=True,
            sensitive_features=sensitive_features,
            control_features=control_features,
        )
        # The following uses X  so that the estimators get X untouched
        super().load_data(X, y_train, sensitive_features=sf_train)
        self._index = [_ALL]

    @property
    def index(self) -> pd.Index:
        """Return the index listing the constraints."""
        return self._index

    def gamma(self, predictor: Callable) -> pd.Series:
        """Return the gamma values for the given predictor."""
        pred = predictor(self.X)
        if isinstance(pred, np.ndarray):
            # TensorFlow is returning an (n,1) array, which results
            # in the subtraction in the 'error =' line generating an
            # (n,n) array
            pred = np.squeeze(pred)
        signed_errors = self.tags[_LABEL] - pred
        total_fn_cost = np.sum(signed_errors[signed_errors > 0] * self.fn_cost)
        total_fp_cost = np.sum(-signed_errors[signed_errors < 0] * self.fp_cost)
        error_value = (total_fn_cost + total_fp_cost) / self.total_samples
        error = pd.Series(data=error_value, index=self.index)
        self._gamma_descr = str(error)
        return error

    def project_lambda(self, lambda_vec: pd.Series) -> pd.Series:
        """Return the lambda values."""
        return lambda_vec

    def signed_weights(self, lambda_vec: pd.Series | None = None) -> pd.Series:
        """Return the signed weights."""
        weights = -self.fp_cost + (self.fp_cost + self.fn_cost) * self.tags[_LABEL]
        if lambda_vec is None:
            return weights
        else:
            return lambda_vec[_ALL] * weights
