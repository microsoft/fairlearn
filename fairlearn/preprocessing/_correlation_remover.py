# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from collections.abc import Iterable

import narwhals.stable.v1 as nw
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from fairlearn.utils._fixes import validate_data


class CorrelationRemover(TransformerMixin, BaseEstimator):
    r"""
    A component that filters out sensitive correlations in a dataset.

    CorrelationRemover applies a linear transformation to the non-sensitive feature columns
    in order to remove their correlation with the sensitive feature columns while retaining
    as much information as possible (as measured by the least-squares error).

    Read more in the :ref:`User Guide <preprocessing>`.

    Parameters
    ----------
        sensitive_feature_ids : list
            list of columns to filter out this can be a sequence of
            either int, in the case of numpy, or string, in the case of pandas.
        alpha : float
            parameter to control how much to filter, for alpha=1.0 we filter out
            all information while for alpha=0.0 we don't apply any.

    Notes
    -----
    This method will change the original dataset by removing all correlation with sensitive
    values. In mathematical terms, assume we have the original dataset :math:`\mathbf{X}`, which
    contains a set of **sensitive features** denoted by :math:`\mathbf{S}` and a set of
    **non-sensitive features** denoted by :math:`\mathbf{Z}`. The goal is to remove
    correlations between the sensitive features and the non-sensitive features.

    Let :math:`m_s` and :math:`m_{ns}` denote the number of sensitive and non-sensitive
    features, respectively.
    Let :math:`\bar{\mathbf{s}}` represent the mean of the sensitive features, *i.e.*,
    :math:`\bar{\mathbf{s}} = [\bar{s}_1, \dots, \bar{s}_{m_s}]^\top`, where
    :math:`\bar{s}_j` is the mean of the :math:`j\text{-th}` sensitive feature.

    For each non-sensitive feature :math:`\mathbf{z}_j\in\mathbb{R}^n`, where
    :math:`j=1,\dotsc,m_{ns}`, we compute an optimal weight vector
    :math:`\mathbf{w}_j^* \in \mathbb{R}^{m_s}` that minimizes the following least squares
    objective:

    .. math::

        \min _{\mathbf{w}} \| \mathbf{z}_j - (\mathbf{S}-\mathbf{1}_n\times\bar{\mathbf{s}}^\top) \mathbf{w} \|_2^2

    where :math:`\mathbf{1}_n` is the all-one vector in :math:`\mathbb{R}^n`.

    In other words, :math:`\mathbf{w}_j^*` is the solution to a linear regression problem where
    we project :math:`\mathbf{z}_j` onto the centered sensitive features. The weight matrix
    :math:`\mathbf{W}^* = (\mathbf{w}_1^*, \dots, \mathbf{w}_{m_{ns}}^*)` is thus obtained by
    solving this regression for each non-sensitive feature.

    Once we have the optimal weight matrix :math:`\mathbf{W}^*`, we compute the **residual
    non-sensitive features** :math:`\mathbf{Z}^*` as follows:

    .. math::

        \mathbf{Z}^* = \mathbf{Z} - (\mathbf{S}-\mathbf{1}_n\times\bar{\mathbf{s}}^\top) \mathbf{W}^*

    The columns in :math:`\mathbf{S}` will be dropped from the dataset :math:`\mathbf{X}`, and
    :math:`\mathbf{Z}^*` will replace the original non-sensitive features :math:`\mathbf{Z}`, but
    the hyper parameter :math:`\alpha` does allow you to tweak the amount of filtering that gets
    applied:

    .. math::

        \mathbf{X}_{\text{tfm}} = \alpha \mathbf{X}_{\text{filtered}} + (1-\alpha) \mathbf{X}_{\text{orig}}

    Note that the lack of correlation does not imply anything about statistical dependence.
    Therefore, we expect this to be most appropriate as a preprocessing step for
    (generalized) linear models.

    .. versionadded:: 0.6

    """

    def __init__(self, *, sensitive_feature_ids: Iterable = (), alpha: float = 1):
        self.sensitive_feature_ids = sensitive_feature_ids
        self.alpha = alpha

    def _split_X(self, X):
        """Split up X into a sensitive and non-sensitive group."""
        sensitive = [self.lookup_[i] for i in self.sensitive_feature_ids]
        non_sensitive = [i for i in range(X.shape[1]) if i not in sensitive]
        return X[:, non_sensitive], X[:, sensitive]

    def _create_lookup(self, X):
        """Create a lookup to handle column names correctly."""
        X = nw.from_native(X, pass_through=True, eager_only=True)
        if isinstance(X, nw.DataFrame):
            self.lookup_ = {c: i for i, c in enumerate(X.columns)}
            return X.to_numpy()
        # correctly handle a 1d input
        X = validate_data(self, X, ensure_2d=False, ensure_min_samples=0)
        if len(X.shape) == 1:
            return {0: 0}
        self.lookup_ = {i: i for i in range(X.shape[1])}
        return X

    def fit(self, X, y=None):
        """Learn the projection required to make the dataset uncorrelated with sensitive columns."""  # noqa: E501

        first_call = not hasattr(self, "_n_features_in_")

        self._check_sensitive_features_in_X(X)
        self._create_lookup(X)
        X = validate_data(self, X)

        if not first_call:
            if self._n_features_in_ != X.shape[1]:
                raise ValueError(
                    "X has %d features, but %s is expecting %d features as input"
                    % (X.shape[1], self.__class__.__name__, self._n_features_in_)
                )

        X_use, X_sensitive = self._split_X(X)

        # correctly handle zero provided sensitive features
        self.sensitive_mean_ = np.array([]) if X_sensitive.shape[1] == 0 else X_sensitive.mean()

        X_s_center = X_sensitive - self.sensitive_mean_
        self.beta_, _, _, _ = np.linalg.lstsq(X_s_center, X_use, rcond=None)

        self._n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """Transform X by applying the correlation remover."""
        check_is_fitted(self, ["beta_", "_n_features_in_", "lookup_", "sensitive_mean_"])

        X = validate_data(self, X)
        if self._n_features_in_ != X.shape[1]:
            raise ValueError(
                "X has %d features, but %s is expecting %d features as input"
                % (X.shape[1], self.__class__.__name__, self._n_features_in_)
            )

        X_use, X_sensitive = self._split_X(X)
        X_s_center = X_sensitive - self.sensitive_mean_
        X_filtered = X_use - X_s_center.dot(self.beta_)
        X_use = np.atleast_2d(X_use)
        X_filtered = np.atleast_2d(X_filtered)
        return self.alpha * X_filtered + (1 - self.alpha) * X_use

    def _check_sensitive_features_in_X(self, X) -> None:
        """Check if the sensitive features are in X."""
        X = nw.from_native(X, pass_through=True, eager_only=True)
        if isinstance(X, nw.DataFrame):
            missing_columns = [c for c in self.sensitive_feature_ids if c not in X.columns]
        else:
            X = validate_data(self, X, ensure_2d=False, ensure_min_samples=0)
            if X.ndim == 1:
                return
            missing_columns = [i for i in self.sensitive_feature_ids if i not in range(X.shape[1])]

        if len(missing_columns) > 0:
            raise ValueError(
                "0 feature(s) (shape=(%d, 0)) while a minimum of %d is required. "
                "Columns %s not found in the input data."
                % (len(missing_columns), len(missing_columns), missing_columns)
            )
