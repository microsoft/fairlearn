# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from fairlearn.utils._input_validation import _validate_and_reformat_input

from .moment import _ALL, _GROUP_ID, _LABEL, _LOSS, _PREDICTION, LossMoment


class ConditionalLossMoment(LossMoment):
    r"""A moment for constraining the mean loss or the worst-case loss by a group.

    Read more in the :ref:`User Guide <bounded_group_loss>`.

    Parameters
    ----------
    loss : {SquareLoss, AbsoluteLoss}
        A loss object with an `eval` method, e.g. `SquareLoss` or
        `AbsoluteLoss`.
    upper_bound : float | None
        An upper bound on the loss, also referred to as :math:`\\zeta`;
        `upper_bound` is an optional argument that is not always
        required; default None
    no_groups : bool
        indicates whether to calculate the mean loss or group-level losses,
        default False, i.e., group-level losses are the default behavior
    """

    def __init__(self, loss, *, upper_bound: float | None = None, no_groups: bool = False):
        super().__init__(loss)
        self.upper_bound = upper_bound
        self.no_groups = no_groups

    @property
    def index(self) -> pd.Index:
        """Return the index listing the constraints."""
        return self._index

    def default_objective(self) -> MeanLoss:
        """Return a default objective."""
        return MeanLoss(self.reduction_loss)

    def load_data(self, X, y, *, sensitive_features) -> None:
        """Load data into the moment object."""
        _, y_train, sf_train, _ = _validate_and_reformat_input(
            X, y, enforce_binary_labels=False, sensitive_features=sensitive_features
        )
        if self.no_groups:
            sf_train = y_train.apply(lambda v: _ALL)

        # The following uses X and not X_train so that the estimators get X untouched
        super().load_data(X, y_train, sensitive_features=sf_train)
        self.prob_attr = self.tags.groupby(_GROUP_ID).size() / self.total_samples
        self._index = self.prob_attr.index
        self.default_objective_lambda_vec = self.prob_attr

        # fill in the information about the basis
        attr_vals = self.tags[_GROUP_ID].unique()
        self.pos_basis = pd.DataFrame()
        self.neg_basis = pd.DataFrame()
        self.neg_basis_present = pd.Series(dtype="float64")
        zero_vec = pd.Series(0.0, self.index)
        i = 0
        for attr in attr_vals:
            self.pos_basis[i] = 0 + zero_vec
            self.neg_basis[i] = 0 + zero_vec
            self.pos_basis.loc[attr, i] = 1
            self.neg_basis_present.at[i] = False
            i += 1

    def gamma(self, predictor: Callable) -> pd.Series:
        """Calculate the degree to which constraints are currently violated by the predictor."""
        self.tags[_PREDICTION] = predictor(self.X)
        self.tags[_LOSS] = self.reduction_loss.eval(self.tags[_LABEL], self.tags[_PREDICTION])
        expect_attr = self.tags.groupby(_GROUP_ID).mean()
        self._gamma_descr = str(expect_attr[[_LOSS]])
        return expect_attr[_LOSS]

    def bound(self) -> pd.Series:
        """Return the vector of bounds.

        Returns
        -------
        pandas.Series
            A vector of bounds on group-level losses
        """
        if self.upper_bound is None:
            raise ValueError("No Upper Bound")
        return pd.Series(self.upper_bound, index=self.index)

    def project_lambda(self, lambda_vec: pd.Series) -> pd.Series:
        """Return the lambda values."""
        return lambda_vec

    def signed_weights(self, lambda_vec: pd.Series | None = None) -> pd.Series:
        """Return the signed weights."""
        if lambda_vec is None:
            adjust = pd.Series(1.0, index=self.index)
        else:
            adjust = lambda_vec / self.prob_attr
        return self.tags.apply(lambda row: adjust[row[_GROUP_ID]], axis=1)


# Ensure that ConditionalLossMoment shows up in correct place in documentation
# when it is used as a base class
ConditionalLossMoment.__module__ = "fairlearn.reductions"


class MeanLoss(ConditionalLossMoment):
    """Moment for evaluating the mean loss.

    Read more in the :ref:`User Guide <constraints_regression>`.
    """

    def __init__(self, loss):
        super().__init__(loss, upper_bound=None, no_groups=True)


class BoundedGroupLoss(ConditionalLossMoment):
    """Moment for constraining the worst-case loss by a group.

    For more information refer to the :ref:`user guide <bounded_group_loss>`.
    """

    def __init__(self, loss, *, upper_bound=None):
        super().__init__(loss, upper_bound=upper_bound, no_groups=False)


class SquareLoss:
    """Class to evaluate the square loss.

    Read more in the :ref:`User Guide <constraints_regression>`.
    """

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        self.min = 0
        self.max = (max_val - min_val) ** 2

    def eval(self, y_true, y_pred):  # noqa: A003
        """Evaluate the square loss for the given set of true and predicted values."""
        return (
            np.clip(y_true, self.min_val, self.max_val)
            - np.clip(y_pred, self.min_val, self.max_val)
        ) ** 2


class AbsoluteLoss:
    """Class to evaluate absolute loss.

    Read more in the :ref:`User Guide <constraints_regression>`.
    """

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        self.min = 0
        self.max = np.abs(max_val - min_val)

    def eval(self, y_true, y_pred):  # noqa: A003
        """Evaluate the absolute loss for the given set of true and predicted values."""
        return np.abs(
            np.clip(y_true, self.min_val, self.max_val)
            - np.clip(y_pred, self.min_val, self.max_val)
        )


# Ensure that AbsoluteLoss shows up in correct place in documentation
# when it is used as a base class
AbsoluteLoss.__module__ = "fairlearn.reductions"


class ZeroOneLoss(AbsoluteLoss):
    """Class to evaluate a zero-one loss.

    Read more in the :ref:`User Guide <constraints_regression>`.
    """

    def __init__(self):
        super().__init__(0, 1)
