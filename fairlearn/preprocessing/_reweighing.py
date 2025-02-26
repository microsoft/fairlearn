# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from ..utils._input_validation import _KW_SENSITIVE_FEATURES, _merge_columns
from numpy import unique, zeros
from numpy import sum as np_sum
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.validation import column_or_1d


"""
Implements the "reweighing" algorithm from
"data preprocessing techniques for classification
without discrimination" paper by Kamiran and Calders.

Reference to paper:
    Kamiran, F., & Calders, T. (2012). Data preprocessing techniques for
    classification without discrimination. Knowledge and information systems,
    33(1), 1-33.
"""


def calculate_weights(Y, A):
    """
    Calculate the weights, according to Alg 3 in the paper.

    Parameters
    ----------
    Y : array
        One-dimensional array-like containing prediction/data to reweight to.
    A : array
        One-dimensional array-like sensitive features.

    Returns
    -------
    W : array
        One-dimensional array-like containing weights
    """
    Y = column_or_1d(check_array(Y, ensure_2d=False, dtype=None))
    A = column_or_1d(_merge_columns(check_array(A, ensure_2d=False, dtype=None)))
    check_consistent_length(Y, A)
    Ys = unique(Y)
    As = unique(A)
    # Per y, store whether an item in Y is y.
    Y_is_y = [(Y == y).astype(int) for y in Ys]
    A_is_a = [(A == a).astype(int) for a in As]
    # Per y, store the sum, because we use it multiple times.
    y_n = [np_sum(is_y) for is_y in Y_is_y]
    a_n = [np_sum(is_a) for is_a in A_is_a]
    W = zeros(len(Y), dtype=float)
    for i in range(len(Ys)):
        for j in range(len(As)):
            # FIXME: laplace smoothing? Or throw meaningful error if denominator
            # is zero?
            is_both = Y_is_y[i] * A_is_a[j]
            w = y_n[i] * a_n[j] / (len(Y) * np_sum(is_both))
            W += w * is_both
    return W


class Reweighing(BaseEstimator, MetaEstimatorMixin):
    """
    Implementation of the reweighing technique from Kamiran and Calders.

    The paper [1]_ by Kamiran and Calders outlines various preprocessing
    techniques. The third one, reweighing, is implemented in this class.
    More specifically, this class is a scikit-learn style meta-estimator
    (wrapper) for Kamiran's algorithm.

    Main functionality is as follows. We wrap this reweighing metaestimator
    (let's call this :code:`MetaEst`) around a trainable scikit-learn style
    estimator (let's call this :code:`Est`). You can then use
    :code:`MetaEst.fit(X, y, sensitive_features)` which will internally compute
    weights :code:`w` using :code:`y` and :code:`sensitive_features`, and pass this
    along using :code:`Est.fit(X, y, sample_weights=w)`.

    NOTE: only works with a single sensitive feature at the moment.

    Parameters
    ----------
    estimator : estimator
        The estimator to wrap.
    passthrough_sensitive_features : bool, default = False
        This boolean determines whether or not to pass the sensitive_features
        through to the estimator.

    References
    ----------
    .. [1] Kamiran, F., & Calders, T. (2012). Data preprocessing techniques for
       classification without discrimination. Knowledge and information systems,
       33(1), 1-33.
    """

    # TODO Possible enhancements:
    # - casting categorical multi-column sensitive feature to single column.
    # - laplace smoothing
    def __init__(self, estimator, *, passthrough_sensitive_features=False):
        self.estimator = estimator
        self.passthrough_sensitive_features = passthrough_sensitive_features

    def fit(self, X, y, **kwargs):
        """
        Fit the model.

        Calculate the weights from :code:`y` and :code:`sensitive_features`,
        and pass the data and the computed weights to the estimator.

        Parameters
        ----------
        X : array-like
            This data is passed directly to the estimator.
        y : 1d array-like
            This data is used to compute weights and passed to the estimator
        sensitive_features : 1d array-like
            This data is used to compute weights and passed to the estimator if
            passthrough_sensitive_features is set to true.

        Returns
        -------
        Object
            returns whatever the estimator returns, which is the estimator
            itself it is following scikit-learn conventions.
        """
        sensitive_features = kwargs.get(_KW_SENSITIVE_FEATURES)
        W = calculate_weights(y, sensitive_features)
        kwargs["sample_weight"] = W
        if self.passthrough_sensitive_features is False:
            del kwargs["sensitive_features"]
        # NOTE: Is returning self better up to sklearn-style?
        return self.estimator.fit(X, y, **kwargs)

    def fit_transform(self, X, y, *, sensitive_features, **kwargs):
        """
        Fit and transform the model.

        Calculate the weights from :code:`y` and :code:`sensitive_features`,
        and pass the data and the computed weights to the estimator.

        Parameters
        ----------
        X : array-like
            This data is passed directly to the estimator.
        y : 1d array-like
            This data is used to compute weights and passed to the estimator
        sensitive_features : 1d array-like
            This data is used to compute weights and passed to the estimator if
            passthrough_sensitive_features is set to true.

        Returns
        -------
        Object
            returns whatever the estimator returns, which is the estimator
            itself it is following scikit-learn conventions.
        """
        sensitive_features = kwargs.get(_KW_SENSITIVE_FEATURES)
        W = calculate_weights(y, sensitive_features)
        kwargs["sample_weight"] = W
        if self.passthrough_sensitive_features is False:
            del kwargs["sensitive_features"]
        # NOTE: Is returning self better up to sklearn-style?
        return self.estimator.fit_transform(X, y, **kwargs)
