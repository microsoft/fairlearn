# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

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
    and false negative errors respectively. The default misclassification
    error corresponds to :math:`c_{FP}=c_{FN}=1.0`.

    Read more in the :ref:`User Guide <error_rate>`.

    Parameters
    ----------
    costs : dict
        Dictionary with keys :code:`'fp'` and :code:`'fn'` containing the
        costs of false positives and false negatives. If none are provided
        costs of 1.0 are assumed.
    """

    short_name = "Err"

    def __init__(self, *, costs=None):
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

    def load_data(self, X, y, *, sensitive_features):
        """Load the specified data into the object.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The feature array
        y : array-like of shape (n_samples,)
            The label vector
        sensitive_features : array-like of shape (n_samples, n_sensitive_features)
            The sensitive feature vector
        """
        _, y_train, sf_train, _ = _validate_and_reformat_input(
            X,
            y,
            enforce_binary_labels=True,
            sensitive_features=sensitive_features,
        )
        # The following uses X so that the estimators get X untouched
        super().load_data(X, y_train, sensitive_features=sf_train)
        self.index = [_ALL]

    def gamma(self, predictor):
        """Calculate the degree to which constraints are currently violated by the
        predictor. The gamma value is normalised by the number of samples.

        Parameters
        ----------
        predictor : func
            bound function returning predictions

        Returns
        -------
        error : :class:`pandas.Series`
            gamma value for the predictor
        """
        y_pred = predictor(self.X)
        if isinstance(y_pred, np.ndarray):
            # TensorFlow is returning an (n,1) array, which results
            # in the subtraction in the 'error =' line generating an
            # (n,n) array
            y_pred = np.squeeze(y_pred)
        signed_errors = self.tags[_LABEL] - y_pred
        total_fn_cost = np.sum(signed_errors[signed_errors > 0] * self.fn_cost)
        total_fp_cost = np.sum(-signed_errors[signed_errors < 0] * self.fp_cost)
        error_value = (total_fn_cost + total_fp_cost) / self.total_samples
        error = pd.Series(data=error_value, index=self.index)
        self._gamma_descr = str(error)
        return error

    def project_lambda(self, lambda_vec):
        """Return the lambda values."""
        return lambda_vec

    def signed_weights(self, lambda_vec=None):
        """Return the signed weights."""
        weights = -self.fp_cost + (self.fp_cost + self.fn_cost) * self.tags[_LABEL]
        if lambda_vec is None:
            return weights
        else:
            return lambda_vec[_ALL] * weights
