# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from fairlearn.utils._input_validation import (
    _MESSAGE_RATIO_NOT_IN_RANGE,
    _validate_and_reformat_input,
)

from .error_rate import ErrorRate
from .moment import _ALL, _EVENT, _GROUP_ID, _LABEL, _SIGN, ClassificationMoment

_UPPER_BOUND_DIFF = "upper_bound_diff"
_LOWER_BOUND_DIFF = "lower_bound_diff"
_MESSAGE_INVALID_BOUNDS = "Only one of difference_bound and ratio_bound can be used."
_DEFAULT_DIFFERENCE_BOUND = 0.01

_CTRL_EVENT_FORMAT = "control={0},{1}"


def _combine_event_and_control(event: str, control: str) -> str:
    if pd.notnull(control):
        return _CTRL_EVENT_FORMAT.format(control, event)
    return event


def _merge_event_and_control_columns(event_col: pd.Series, control_col) -> pd.Series:
    if control_col is None:
        return event_col
    return event_col.combine(control_col, _combine_event_and_control)


class UtilityParity(ClassificationMoment):
    r"""A generic moment for parity in utilities (or costs) under classification.

    This serves as the base class for :class:`DemographicParity`,
    :class:`EqualizedOdds`, and others. All subclasses can be used as
    difference-based constraints or ratio-based constraints.
    Refer to the :ref:`user guide <constraints_binary_classification>` for
    more information and example usage.

    Constraints compare the group-level mean utility for each group with the
    overall mean utility (unless further events are specified, e.g., in
    equalized odds).
    Constraint violation for difference-based constraints starts if the
    difference between a group and the overall population with regard to a
    utility exceeds `difference_bound`. For ratio-based constraints, the ratio
    between the group-level and overall mean utility needs to be bounded
    between `ratio_bound` and its inverse (plus an additional additive
    `ratio_bound_slack`).

    The `index` field is a :class:`pandas:pandas.MultiIndex` corresponding to
    the constraint IDs.
    It is an index of various DataFrame and Series objects that are either
    required as arguments or returned by several of the methods of the
    `UtilityParity` class. It is the Cartesian product of:

    - The unique events defining the particular moment object
    - The unique values of the sensitive feature
    - The characters `+` and `-`, corresponding to the Lagrange multipliers
      for positive and negative violations of the constraint

    Read more in the :ref:`User Guide <constraints_binary_classification>`.

    Parameters
    ----------
    difference_bound : float | None
        The constraints' difference bound for constraints that are expressed
        as differences, also referred to as :math:`\\epsilon` in documentation.
        If `ratio_bound` is used then `difference_bound` needs to be None.
        If neither `ratio_bound` nor `difference_bound` are set then a default
        difference bound of 0.01 is used for backwards compatibility.
        Default None.
    ratio_bound : float | None
        The constraints' ratio bound for constraints that are expressed as
        ratios. The specified value needs to be in (0,1].
        If `difference_bound` is used then `ratio_bound` needs to be None.
        Default None.
    ratio_bound_slack : float
        The constraints' ratio bound slack for constraints that are
        expressed as ratios, also referred to as :math:`\\epsilon` in
        documentation.
        `ratio_bound_slack` is ignored if `ratio_bound` is not specified.
        Default 0.0
    """

    def __init__(
        self,
        *,
        difference_bound: float | None = None,
        ratio_bound: float | None = None,
        ratio_bound_slack: float = 0.0,
    ):
        """Initialize with the ratio value."""
        super(UtilityParity, self).__init__()
        if (difference_bound is None) and (ratio_bound is None):
            self.eps = _DEFAULT_DIFFERENCE_BOUND
            self.ratio = 1.0
        elif (difference_bound is not None) and (ratio_bound is None):
            self.eps = difference_bound
            self.ratio = 1.0
        elif (difference_bound is None) and (ratio_bound is not None):
            self.eps = ratio_bound_slack
            if not (0 < ratio_bound <= 1):
                raise ValueError(_MESSAGE_RATIO_NOT_IN_RANGE)
            self.ratio = ratio_bound
        else:
            # both difference_bound and ratio_bound specified
            raise ValueError(_MESSAGE_INVALID_BOUNDS)

    @property
    def index(self) -> pd.MultiIndex:
        """Return the multi-index listing the constraints."""
        return self._index

    def default_objective(self) -> ErrorRate:
        """Return the default objective for moments of this kind."""
        return ErrorRate()

    def load_data(
        self,
        X,
        y: pd.Series,
        *,
        sensitive_features: pd.Series,
        event: pd.Series | None = None,
        utilities=None,
    ) -> None:
        """Load the specified data into this object.

        This adds a column `event` to the `tags` field.

        The `utilities` is a 2-d array which corresponds to g(X,A,Y,h(X))
        from :footcite:t:`agarwal2018reductions`.
        The `utilities` defaults to h(X), i.e. [0, 1] for each X_i.
        The first column is G^0 and the second is G^1.
        Assumes binary classification with labels 0/1.

        .. math::
            utilities = [g(X,A,Y,h(X)=0), g(X,A,Y,h(X)=1)]

        """
        super().load_data(X, y, sensitive_features=sensitive_features)
        self.tags[_EVENT] = event
        if utilities is None:
            utilities = np.vstack(
                [
                    np.zeros(y.shape, dtype=np.float64),
                    np.ones(y.shape, dtype=np.float64),
                ]
            ).T
        self.utilities = utilities
        self.utility_diff = self.utilities[:, 1] - self.utilities[:, 0]
        self.prob_event = self.tags.groupby(_EVENT).size() / self.total_samples
        self.prob_group_event = self.tags.groupby([_EVENT, _GROUP_ID]).size() / self.total_samples
        signed = pd.concat(
            [self.prob_group_event, self.prob_group_event],
            keys=["+", "-"],
            names=[_SIGN, _EVENT, _GROUP_ID],
        )
        self._index = signed.index
        self.default_objective_lambda_vec = None

        # Fill information about the matrix U, which is used to calculate signed weights
        # and gamma vector. The matrix has a row corresponding to each example and column
        # corresponding to each component of lambda, the signed weights and gamma are
        # calculated as:
        #
        # signed_weights = utility_diff * U.dot(lambda_vec)
        # gamma = -U.T.dot(utilities[y_pred]) / total_samples
        #
        # If the i-th example's event and group are "e" and "g" then the entries in the
        # i-th row of the matrix, indexed by tuples (sign, event, group), are defined as:
        #
        # U_i,(+,e',g') =      1[e=e']/P(e) - rho*1[e=e',g=g']/P(g,e)
        # U_i,(-,e',g') = -rho*1[e=e']/P(e) +     1[e=e',g=g']/P(g,e)
        #
        # This corresponds to the expression given right above Theorem 1 of the paper
        # "A Reductions Approach to Fair Classification" by Agarwal et al. (2018).

        self.U = pd.DataFrame(0, index=self.tags.index, columns=self.index)
        for e, g in self.prob_group_event.index:
            event_select = 1 * (self.tags[_EVENT] == e)
            group_event_select = event_select * (self.tags[_GROUP_ID] == g)
            self.U["+", e, g] = (
                event_select / self.prob_event[e]
                + (-self.ratio) * group_event_select / self.prob_group_event[e, g]
            )
            self.U["-", e, g] = (-self.ratio) * event_select / self.prob_event[
                e
            ] + group_event_select / self.prob_group_event[e, g]

        # fill in the information about the basis
        event_vals = self.tags[_EVENT].dropna().unique()
        group_vals = self.tags[_GROUP_ID].unique()
        # The matrices pos_basis and neg_basis contain a lower-dimensional description of
        # constraints, which is achieved by removing some redundant constraints.
        # Considering fewer constraints is not required for correctness, but it can dramatically
        # speed up GridSearch.
        self.neg_basis_present = pd.Series(dtype="float64")
        col_count = len(event_vals) * (len(group_vals) - 1)
        self.pos_basis = pd.DataFrame(0.0, index=self.index, columns=range(col_count)).sort_index()
        self.neg_basis = pd.DataFrame(0.0, index=self.index, columns=range(col_count)).sort_index()

        i = 0

        for e in event_vals:
            # Constraints on the final group are redundant, so they are not
            # included in the basis.
            for g in group_vals[:-1]:
                if ("+", e, g) in self.index:
                    self.pos_basis.loc[("+", e, g), i] = 1
                if ("-", e, g) in self.index:
                    self.neg_basis.loc[("-", e, g), i] = 1
                self.neg_basis_present.at[i] = True
                i += 1

    def gamma(self, predictor: Callable) -> pd.Series:
        """Calculate the degree to which constraints are currently violated by the predictor."""
        predictions = predictor(self.X)
        if isinstance(predictions, np.ndarray):
            # TensorFlow seems to return an (n,1) array instead of an (n) array
            predictions = np.squeeze(predictions)
        pred = self.utility_diff.T * predictions + self.utilities[:, 0]
        g_signed = -self.U.T.dot(pred) / self.total_samples
        self._gamma_descr = str(g_signed)
        return g_signed

    def bound(self) -> pd.Series:
        """Return bound vector.

        Returns
        -------
        pandas.Series
            a vector of bound values corresponding to all constraints

        """
        return pd.Series(self.eps, index=self.index)

    # TODO: this can be further improved using the overcompleteness in group membership
    def project_lambda(self, lambda_vec: pd.Series) -> pd.Series:
        """Return the projected lambda values.

        i.e., returns lambda which is guaranteed to lead to the same or higher value of the
        Lagrangian compared with lambda_vec for all possible choices of the classifier, h.
        """
        if self.ratio == 1.0:
            lambda_pos = lambda_vec["+"] - lambda_vec["-"]
            lambda_neg = -lambda_pos
            lambda_pos[lambda_pos < 0.0] = 0.0
            lambda_neg[lambda_neg < 0.0] = 0.0
            lambda_projected = pd.concat(
                [lambda_pos, lambda_neg],
                keys=["+", "-"],
                names=[_SIGN, _EVENT, _GROUP_ID],
            )
            return lambda_projected
        return lambda_vec

    def signed_weights(self, lambda_vec: pd.Series) -> pd.Series:
        """Compute the signed weights.

        Uses the equations for :math:`C_i^0` and :math:`C_i^1` as defined
        in Section 3.2 of :footcite:t:`agarwal2018reductions`
        in the 'best response of the Q-player' subsection to compute the
        signed weights to be applied to the data by the next call to the underlying
        estimator.

        Parameters
        ----------
        lambda_vec : :class:`pandas:pandas.Series`
            The vector of Lagrange multipliers indexed by `index`

        """
        return self.utility_diff * self.U.dot(lambda_vec)


# Ensure that UtilityParity shows up in correct place in documentation
# when it is used as a base class
UtilityParity.__module__ = "fairlearn.reductions"


class DemographicParity(UtilityParity):
    r"""Implementation of demographic parity as a moment.

    A classifier :math:`h(X)` satisfies demographic parity if

    .. math::
      P[h(X) = 1 | A = a] = P[h(X) = 1] \; \forall a

    This implementation of :class:`UtilityParity` defines
    a single event, `all`. Consequently, the `prob_event`
    :class:`pandas:pandas.Series`
    will only have a single entry, which will be equal to 1.
    Similarly, the `index` property will have twice as many entries
    (corresponding to the Lagrange multipliers for positive and negative
    constraints) as there are unique values for the sensitive feature.
    The :meth:`UtilityParity.signed_weights` method will compute the costs
    according to Example 3 of :footcite:t:`agarwal2018reductions`.

    This :class:`~Moment` also supports control features, which can be used to
    stratify the data, with the Demographic Parity constraint applied within
    each stratum, but not between strata. If the control feature groups
    are :math:`c \in \mathcal{C}` then the above equation will become

    .. math::
      P[h(X) = 1 | A = a, C = c] = P[h(X) = 1 | C = c] \; \forall a, c

    Read more in the :ref:`User Guide <demographic_parity>`.

    """

    short_name = "DemographicParity"

    def load_data(self, X, y, *, sensitive_features, control_features=None) -> None:
        """Load the specified data into the object."""
        _, y_train, sf_train, cf_train = _validate_and_reformat_input(
            X,
            y,
            enforce_binary_labels=True,
            sensitive_features=sensitive_features,
            control_features=control_features,
        )

        base_event = pd.Series(data=_ALL, index=y_train.index)
        event = _merge_event_and_control_columns(base_event, cf_train)
        super().load_data(X, y_train, event=event, sensitive_features=sf_train)


class TruePositiveRateParity(UtilityParity):
    r"""Implementation of true positive rate parity as a moment.

    .. note::

        The true positive rate parity fairness criterion is also known
        as "equal opportunity".

    Adds conditioning on label `Y=1` compared to demographic parity, i.e.,

    .. math::
       P[h(X) = 1 | A = a, Y = 1] = P[h(X) = 1 | Y = 1] \; \forall a

    This implementation of :class:`UtilityParity` defines the event
    corresponding to `Y=1`.

    The `prob_event` :class:`pandas:pandas.DataFrame` will record the fraction
    of the samples corresponding to `Y = 1` in the `Y` array.

    The `index` MultiIndex will have a number of entries equal to the number
    of unique values of the sensitive feature, multiplied by the number of
    unique non-NaN values of the constructed `event` array, whose entries
    are either NaN or `label=1` (so only one unique non-NaN value), multiplied
    by two (for the Lagrange multipliers for positive and negative
    constraints).

    With these definitions, the :meth:`UtilityParity.signed_weights` method
    will calculate the costs for `Y=1` as they are calculated in Example 4 of
    :footcite:t:`agarwal2018reductions`, but will use
    the weights equal to zero for `Y=0`.

    This :class:`~Moment` also supports control features, which can be used to
    stratify the data, with the constraint applied within
    each stratum, but not between strata.

    Read more in the :ref:`User Guide <true_positive_rate_parity>`.

    """

    short_name = "TruePositiveRateParity"

    def load_data(self, X, y, *, sensitive_features, control_features=None) -> None:
        """Load the specified data into the object."""
        _, y_train, sf_train, cf_train = _validate_and_reformat_input(
            X,
            y,
            enforce_binary_labels=True,
            sensitive_features=sensitive_features,
            control_features=control_features,
        )

        # The `where` clause is used to put `pd.nan` on all values where `Y!=1`.
        base_event = y_train.apply(lambda v: _LABEL + "=" + str(v)).where(y_train == 1)
        event = _merge_event_and_control_columns(base_event, cf_train)
        super().load_data(X, y_train, event=event, sensitive_features=sf_train)


class FalsePositiveRateParity(UtilityParity):
    r"""Implementation of false positive rate parity as a moment.

    Adds conditioning on label `Y=0` compared to demographic parity, i.e.,

    .. math::
       P[h(X) = 1 | A = a, Y = 0] = P[h(X) = 1 | Y = 0] \; \forall a

    This implementation of :class:`UtilityParity` defines the event
    corresponding to `Y=0`.

    The `prob_event` :class:`pandas:pandas.DataFrame` will record the fraction
    of the samples corresponding to `Y = 0` in the `Y` array.

    The `index` MultiIndex will have a number of entries equal to the number
    of unique values of the sensitive feature, multiplied by the number of
    unique non-NaN values of the constructed `event` array, whose entries are
    either NaN or `label=0` (so only one unique non-NaN value), multiplied by
    two (for the Lagrange multipliers for positive and negative constraints).

    With these definitions, the :meth:`UtilityParity.signed_weights` method will calculate
    the costs for `Y=0` as they are calculated in Example 4 of
    :footcite:t:`agarwal2018reductions`,
    but will use the weights equal to zero for `Y=1`.

    This :class:`~Moment` also supports control features, which can be used to
    stratify the data, with the constraint applied within
    each stratum, but not between strata.

    Read more in the :ref:`User Guide <false_positive_rate_parity>`.

    """

    short_name = "FalsePositiveRateParity"

    def load_data(self, X, y, *, sensitive_features, control_features=None) -> None:
        """Load the specified data into the object."""
        _, y_train, sf_train, cf_train = _validate_and_reformat_input(
            X,
            y,
            enforce_binary_labels=True,
            sensitive_features=sensitive_features,
            control_features=control_features,
        )

        # The `where` clause is used to put `pd.nan` on all values where `Y!=0`.
        base_event = y_train.apply(lambda v: _LABEL + "=" + str(v)).where(y_train == 0)
        event = _merge_event_and_control_columns(base_event, cf_train)
        super().load_data(X, y_train, event=event, sensitive_features=sf_train)


class EqualizedOdds(UtilityParity):
    r"""Implementation of equalized odds as a moment.

    Adds conditioning on label compared to demographic parity, i.e.

    .. math::
       P[h(X) = 1 | A = a, Y = y] = P[h(X) = 1 | Y = y] \; \forall a, y

    This implementation of :class:`UtilityParity` defines
    events corresponding to the unique values of the `Y` array.

    The `prob_event` :class:`pandas:pandas.Series` will record the
    fraction of the samples corresponding to each unique value in
    the `Y` array.

    The `index` MultiIndex will have a number of entries equal to
    the number of unique values for the sensitive feature, multiplied by
    the number of unique values of the `Y` array, multiplied by two (for
    the Lagrange multipliers for positive and negative constraints).

    With these definitions, the :meth:`UtilityParity.signed_weights` method
    will calculate the costs according to Example 4 of
    :footcite:t:`agarwal2018reductions`.

    This :class:`~Moment` also supports control features, which can be used to
    stratify the data, with the constraint applied within
    each stratum, but not between strata.

    Read more in the :ref:`User Guide <equalized_odds>`.

    """

    short_name = "EqualizedOdds"

    def load_data(self, X, y, *, sensitive_features, control_features=None) -> None:
        """Load the specified data into the object."""
        _, y_train, sf_train, cf_train = _validate_and_reformat_input(
            X,
            y,
            enforce_binary_labels=True,
            sensitive_features=sensitive_features,
            control_features=control_features,
        )

        base_event = y_train.apply(lambda v: _LABEL + "=" + str(v))
        event = _merge_event_and_control_columns(base_event, cf_train)
        super().load_data(X, y_train, event=event, sensitive_features=sf_train)


class ErrorRateParity(UtilityParity):
    r"""Implementation of error rate parity as a moment.

    A classifier :math:`h(X)` satisfies error rate parity if

    .. math::
      P[h(X) \ne Y | A = a] = P[h(X) \ne Y] \; \forall a

    This implementation of :class:`UtilityParity` defines
    a single event, `all`. Consequently, the `prob_event`
    :class:`pandas:pandas.Series` will only have a single
    entry, which will be equal to 1.

    The `index` property will have twice as many entries (corresponding to the Lagrange multipliers
    for positive and negative constraints) as there are unique values for the sensitive feature.

    The :meth:`UtilityParity.signed_weights` method will compute the costs according to Example 3
    of :footcite:t:`agarwal2018reductions`.
    However, in this scenario, g = abs(h(x)-y), rather than g = h(x)

    This :class:`~Moment` also supports control features, which can be used to
    stratify the data, with the constraint applied within
    each stratum, but not between strata.

    Read more in the :ref:`User Guide <error_rate_parity>`.

    """

    short_name = "ErrorRateParity"

    def load_data(self, X, y, *, sensitive_features, control_features=None) -> None:
        """Load the specified data into the object."""
        _, y_train, sf_train, cf_train = _validate_and_reformat_input(
            X,
            y,
            enforce_binary_labels=True,
            sensitive_features=sensitive_features,
            control_features=control_features,
        )
        utilities = np.vstack([y_train, 1 - y_train]).T
        base_event = pd.Series(data=_ALL, index=y_train.index)
        event = _merge_event_and_control_columns(base_event, cf_train)
        super().load_data(
            X, y_train, event=event, utilities=utilities, sensitive_features=sf_train
        )
