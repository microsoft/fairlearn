.. _reductions:

Reductions
==========

.. currentmodule:: fairlearn.reductions

On a high level, the reduction algorithms within Fairlearn
enable unfairness mitigation for an arbitrary machine learning model with
respect to user-provided fairness constraints. All of the constraints currently supported
by reduction algorithms are group-fairness constraints. For more information on the
supported fairness constraints refer to :ref:`constraints_binary_classification`
and :ref:`constraints_regression`.

.. note::

   The choice of a fairness metric and fairness constraints is a crucial
   step in the AI development and deployment, and
   choosing an unsuitable constraint can lead to more harms.
   For a broader discussion of fairness as a
   sociotechnical challenge and how to view Fairlearn in this context refer to
   :ref:`fairness_in_machine_learning`.

The reductions approach for classification seeks to reduce binary
classification subject to fairness constraints to a sequence of weighted
classification problems (see :footcite:`agarwal2018reductions`), and similarly for regression (see :footcite:`agarwal2019fair`).
As a result, the reduction algorithms
in Fairlearn only require a wrapper access to any "base" learning algorithm.
By this we mean that the "base" algorithm only needs to implement :code:`fit` and
:code:`predict` methods, as any standard scikit-learn estimator, but it
does not need to have any knowledge of the desired fairness constraints or sensitive features.

From an API perspective this looks as follows in all situations

>>> reduction = Reduction(base_estimator, constraints, **kwargs)  # doctest: +SKIP
>>> reduction.fit(X_train, y_train, sensitive_features=sensitive_features)  # doctest: +SKIP
>>> reduction.predict(X_test)  # doctest: +SKIP

Fairlearn doesn't impose restrictions on the referenced :code:`base_estimator`
other than the existence of :code:`fit` and :code:`predict` methods.
At the moment, the :code:`base_estimator`'s :code:`fit` method also needs to
provide a :code:`sample_weight` argument which the reductions techniques use
to reweight samples.
In the future Fairlearn will provide functionality to handle this even
without a :code:`sample_weight` argument.

Before looking more into reduction algorithms, this section
reviews the supported fairness constraints. All of them
are expressed as objects inheriting from the base class :code:`Moment`.
:code:`Moment`'s main purpose is to calculate the constraint violation of a
current set of predictions through its :code:`gamma` function as well as to
provide :code:`signed_weights` that are used to relabel and reweight samples.

.. _constraints_binary_classification:

Fairness constraints for binary classification
----------------------------------------------

All supported fairness constraints for binary classification inherit from
:code:`UtilityParity`. They are based on some underlying metric called
*utility*, which can be evaluated on individual data points and is averaged
over various groups of data points to form the *utility parity* constraint
of the form

.. math::

    \text{utility}_{a,e} = \text{utility}_e \quad \forall a, e

where :math:`a` is a sensitive feature value and :math:`e` is an *event*
identifier. Each data point has only one value of a sensitive feature,
and belongs to at most one event. In many examples, there is only
a single event :math:`*`, which includes all the data points. Other
examples of events include :math:`Y=0` and :math:`Y=1`. The utility
parity requires that the mean utility within each event equals
the mean utility of each group whose sensitive feature is :math:`a`
within that event.

The class :code:`UtilityParity` implements constraints that allow
some amount of violation of the utility parity constraints, where
the maximum allowed violation is specified either as a difference
or a ratio.

The *difference-based relaxation* starts out by representing
the utility parity constraints as pairs of
inequalities

.. math::

    \text{utility}_{a,e} - \text{utility}_{e} \leq 0 \quad \forall a, e\\
    -\text{utility}_{a,e} + \text{utility}_{e} \leq 0 \quad \forall a, e

and then replaces zero on the right-hand side
with a value specified as :code:`difference_bound`. The resulting
constraints are instantiated as

    >>> UtilityParity(difference_bound=0.01)  # doctest: +SKIP

Note that satisfying these constraints does not mean
that the difference between the groups with the highest and
smallest utility in each event is bounded by :code:`difference_bound`.
The value of :code:`difference_bound` instead bounds
the difference between the utility of each group and the overall mean
utility within each event. This, however,
implies that the difference between groups in each event is
at most twice the value of :code:`difference_bound`.

The *ratio-based relaxation* relaxes the parity
constraint as

.. math::

    r \leq \dfrac{\text{utility}_{a,e}}{\text{utility}_e} \leq \dfrac{1}{r} \quad \forall a, e

for some value of :math:`r` in (0,1]. For example, if :math:`r=0.9`, this means
that within each event
:math:`0.9 \cdot \text{utility}_{a,e} \leq \text{utility}_e`, i.e., the utility for
each group needs to be at least 90% of the overall utility for the event, and
:math:`0.9 \cdot \text{utility}_e \leq \text{utility}_{a,e}`, i.e., the overall utility
for the event needs to be at least 90% of each group's utility.

The two ratio constraints can be rewritten as

.. math::

   - \text{utility}_{a,e} + r \cdot \text{utility}_e \leq 0 \quad \forall a, e \\
   r \cdot \text{utility}_{a,e} - \text{utility}_e \leq 0 \quad \forall a, e

When instantiating the ratio constraints, we use :code:`ratio_bound` for :math:`r`,
and also allow further relaxation by replacing the zeros on the right hand side
by some non-negative :code:`ratio_bound_slack`. The resulting instantiation
looks as

    >>> UtilityParity(ratio_bound=0.9, ratio_bound_slack=0.01)  # doctest: +SKIP

Similarly to the difference constraints, the ratio constraints do not directly
bound the ratio between the pairs of groups, but such a bound is implied.

.. note::

    It is not possible to specify both :code:`difference_bound` *and*
    :code:`ratio_bound` for the same constraint object.

.. _demographic_parity:

Demographic Parity
~~~~~~~~~~~~~~~~~~

A binary classifier :math:`h(X)` satisfies *demographic parity* if

.. math::

    \P[h(X) = 1 \given A = a] = \P[h(X) = 1] \quad \forall a

In other words, the selection rate or percentage of samples with label 1
should be equal across all groups. Implicitly this means the percentage
with label 0 is equal as well. In this case, the utility function
is equal to :math:`h(X)` and there is only a single event :math:`*`.

In the example below group :code:`"a"` has a selection rate of 60%,
:code:`"b"` has a selection rate of 20%. The overall selection rate is 40%,
so :code:`"a"` is `0.2` above the overall selection rate, and :code:`"b"` is
`0.2` below. Invoking the method :code:`gamma` shows the values
of the left-hand sides of the constraints described
in :ref:`constraints_binary_classification`, which is independent
of the provided :code:`difference_bound`. Note that the left-hand sides
corresponding to different values of :code:`sign` are just negatives
of each other.
The value of :code:`y_true` is in this example irrelevant to the calculations,
because the underlying utility in demographic parity, selection rate, does not
consider performance relative to the true labels, but rather proportions in
the predicted labels.

.. note::

    When providing :code:`DemographicParity` to mitigation algorithms, only use
    the constructor and the mitigation algorithm itself then invokes :code:`load_data`.
    The example below uses :code:`load_data` to illustrate how :code:`DemographicParity`
    instantiates inequalities from :ref:`constraints_binary_classification`.

.. doctest:: mitigation_reductions
    :options:  +NORMALIZE_WHITESPACE

    >>> from fairlearn.reductions import DemographicParity
    >>> from fairlearn.metrics import MetricFrame, selection_rate
    >>> import numpy as np
    >>> import pandas as pd
    >>> dp = DemographicParity(difference_bound=0.01)
    >>> X                  = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
    >>> y_true             = np.array([ 1 ,  1 ,  1 ,  1 ,  0,   0 ,  0 ,  0 ,  0 ,  0 ])
    >>> y_pred             = np.array([ 1 ,  1 ,  1 ,  1 ,  0,   0 ,  0 ,  0 ,  0 ,  0 ])
    >>> sensitive_features = np.array(["a", "b", "a", "a", "b", "a", "b", "b", "a", "b"])
    >>> selection_rate_summary = MetricFrame(metrics=selection_rate,
    ...                                      y_true=y_true,
    ...                                      y_pred=y_pred,
    ...                                      sensitive_features=pd.Series(sensitive_features, name="SF 0"))
    >>> selection_rate_summary.overall.item()
    0.4
    >>> selection_rate_summary.by_group
    SF 0
    a    0.6
    b    0.2
    Name: selection_rate, dtype: float64
    >>> dp.load_data(X, y_true, sensitive_features=sensitive_features)
    >>> dp.gamma(lambda X: y_pred)
    sign  event  group_id
    +     all    a           0.2
                 b          -0.2
    -     all    a          -0.2
                 b           0.2
    dtype: float64

The ratio constraints for the demographic parity with :code:`ratio_bound`
:math:`r` (and :code:`ratio_bound_slack=0`) take form

.. math::

    r \leq \dfrac{\P[h(X) = 1 \given A = a]}{\P[h(X) = 1]} \leq \dfrac{1}{r} \quad \forall a

Revisiting the same example as above we get

.. doctest:: mitigation_reductions

    >>> dp = DemographicParity(ratio_bound=0.9, ratio_bound_slack=0.01)
    >>> dp.load_data(X, y_pred, sensitive_features=sensitive_features)
    >>> dp.gamma(lambda X: y_pred)
    sign  event  group_id
    +     all    a           0.14
                 b          -0.22
    -     all    a          -0.24
                 b           0.16
    dtype: float64

Following the expressions for the left-hand sides
of the constraints, we obtain

.. math::

    r \cdot \text{utility}_{a,*} - \text{utility}_* = 0.9 \times 0.6 - 0.4 = 0.14 \\
    r \cdot \text{utility}_{b,*} - \text{utility}_* = 0.9 \times 0.2 - 0.4 = -0.22 \\
    - \text{utility}_{a,*} + r \cdot \text{utility}_* = - 0.6 + 0.9 \times 0.4 = -0.24 \\
    - \text{utility}_{b,*} + r \cdot \text{utility}_* = - 0.2 + 0.9 \times 0.4 = 0.16 \\

.. _true_positive_rate_parity:
.. _false_positive_rate_parity:

True Positive Rate Parity and False Positive Rate Parity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A binary classifier :math:`h(X)` satisfies *true positive rate parity* if

.. math::

    \P[h(X) = 1 \given A = a, Y = 1] = \P[h(X) = 1 \given Y = 1] \quad \forall a

and *false positive rate parity* if

.. math::

    \P[h(X) = 1 \given A = a, Y = 0] = \P[h(X) = 1 \given Y = 0] \quad \forall a

In first case, we only have one event :math:`Y=1` and
ignore the samples with :math:`Y=0`, and in the second case vice versa.
Refer to :ref:`equalized_odds` for the fairness constraint type that simultaneously
enforce both true positive rate parity and false positive rate parity
by considering both events :math:`Y=0` and :math:`Y=1`.

In practice this can be used in a difference-based relaxation as follows:

.. doctest:: mitigation_reductions
    :options:  +NORMALIZE_WHITESPACE

    >>> from fairlearn.reductions import TruePositiveRateParity
    >>> from fairlearn.metrics import true_positive_rate
    >>> import numpy as np
    >>> tprp = TruePositiveRateParity(difference_bound=0.01)
    >>> X                  = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
    >>> y_true             = np.array([ 1 ,  1 ,  1 ,  1 ,  1,   1 ,  1 ,  0 ,  0 ,  0 ])
    >>> y_pred             = np.array([ 1 ,  1 ,  1 ,  1 ,  0,   0 ,  0 ,  1 ,  0 ,  0 ])
    >>> sensitive_features = np.array(["a", "b", "a", "a", "b", "a", "b", "b", "a", "b"])
    >>> tpr_summary = MetricFrame(metrics=true_positive_rate,
    ...                           y_true=y_true,
    ...                           y_pred=y_pred,
    ...                           sensitive_features=sensitive_features)
    >>> tpr_summary.overall.item()
    0.5714285714285714
    >>> tpr_summary.by_group
    sensitive_feature_0
    a    0.75...
    b    0.33...
    Name: true_positive_rate, dtype: float64
    >>> tprp.load_data(X, y_true, sensitive_features=sensitive_features)
    >>> tprp.gamma(lambda X: y_pred)
    sign  event    group_id
    +     label=1  a           0.1785...
                   b          -0.2380...
    -     label=1  a          -0.1785...
                   b           0.2380...
    dtype: float64

.. note::

    When providing :code:`TruePositiveRateParity` or :code:`FalsePositiveRateParity`
    to mitigation algorithms, only use
    the constructor. The mitigation algorithm itself then invokes :code:`load_data`.
    The example uses :code:`load_data` to illustrate how :code:`TruePositiveRateParity`
    instantiates inequalities from :ref:`constraints_binary_classification`.

Alternatively, a ratio-based relaxation is also available:

.. doctest:: mitigation_reductions

    >>> tprp = TruePositiveRateParity(ratio_bound=0.9, ratio_bound_slack=0.01)
    >>> tprp.load_data(X, y_true, sensitive_features=sensitive_features)
    >>> tprp.gamma(lambda X: y_pred)
    sign  event    group_id
    +     label=1  a           0.1035...
                   b          -0.2714...
    -     label=1  a          -0.2357...
                   b           0.1809...
    dtype: float64

.. _equalized_odds:

Equalized Odds
~~~~~~~~~~~~~~

A binary classifier :math:`h(X)` satisfies *equalized odds* if it satisfies both
*true positive rate parity* and *false positive rate parity*, i.e.,

.. math::

    \P[h(X) = 1 \given A = a, Y = y] = \P[h(X) = 1 \given Y = y] \quad \forall a, y

The constraints represent the union of constraints for true positive rate
and false positive rate.

.. doctest:: mitigation_reductions

    >>> from fairlearn.reductions import EqualizedOdds
    >>> eo = EqualizedOdds(difference_bound=0.01)
    >>> eo.load_data(X, y_true, sensitive_features=sensitive_features)
    >>> eo.gamma(lambda X: y_pred)
    sign  event    group_id
    +     label=0  a          -0.3333...
                   b           0.1666...
          label=1  a           0.1785...
                   b          -0.2380...
    -     label=0  a           0.3333...
                   b          -0.1666...
          label=1  a          -0.1785...
                   b           0.2380...
    dtype: float64


.. _error_rate:

Error Rate
~~~~~~~~~~

We can use the :class:`ErrorRate` either to measure performance of a trained model that does or
does not take sensitive features into account, or as an addition to the
objective function during the training of a fairness aware model.

To measure the ErrorRate in respect to a trained estimator we use its :code:`gamma` method:

.. doctest:: mitigation_reductions

    >>> from fairlearn.reductions import ErrorRate
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import LogisticRegression
    >>> import numpy as np
    >>> rng = np.random.default_rng(42) 
    >>> X, y = make_classification(n_features=10, class_sep=0.1, random_state=42)
    >>> X[:, -1] = rng.integers(0, 2, size=(X.shape[0],)) # defining the sensitive feature
    >>> sensitive_features = X[:, -1]
    >>> classifier = LogisticRegression().fit(X, y)
    >>> costs = {"fp":0.1, "fn":0.9}
    >>> errorrate = ErrorRate(costs=costs)
    >>> errorrate.load_data(X, y, sensitive_features=sensitive_features)
    >>> errorrate.gamma(classifier.predict)
    all    0.139
    dtype: float64


Using :class:`ErrorRate` as part of a reductions approach for fairness mitigation is equivalent
to a cost-sensitive classification problem.

.. doctest:: mitigation_reductions

    >>> from fairlearn.reductions import ErrorRate, EqualizedOdds, ExponentiatedGradient
    >>> from fairlearn.metrics import MetricFrame
    >>> from sklearn.metrics import accuracy_score
    >>> import numpy as np
    >>> rng = np.random.default_rng(42) 
    >>> X, y = make_classification(n_features=10, class_sep=0.1, random_state=42)
    >>> X[:, -1] = rng.integers(0, 2, size=(X.shape[0],)) # defining the sensitive feature
    >>> sensitive_features = X[:, -1]
    >>> objective = ErrorRate(costs={"fp":0.1, "fn":0.9})
    >>> constraint = EqualizedOdds(difference_bound=0.01)
    >>> classifier = LogisticRegression()
    >>> mitigator = ExponentiatedGradient(classifier, constraint, objective=objective)
    >>> X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
    ... X, y, sensitive_features, test_size=0.33, random_state=42)
    >>> mitigator.fit(X_train, y_train, sensitive_features=sensitive_train) # doctest: +ELLIPSIS
    ExponentiatedGradient(...)
    >>> y_pred = mitigator.predict(X_test)
    >>> mf_mitigated = MetricFrame(metrics=accuracy_score, y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_test)
    >>> mf_mitigated.overall.item()
    0.5151515151515151
    >>> mf_mitigated.by_group
    sensitive_feature_0
    0.0    0.611111
    1.0    0.400000
    Name: accuracy_score, dtype: float64


.. _error_rate_parity:

Error Rate Parity
~~~~~~~~~~~~~~~~~

The *error rate parity* requires that the error rates should be
the same across all groups. For a classifier :math:`h(X)`
this means that

.. math::

   \P[h(X) \ne Y \given A = a] = \P[h(X) \ne Y] \quad \forall a

In this case, the utility is equal to 1 if :math:`h(X)\ne Y` and equal to
0 if :math:`h(X)=Y`, and so large value of utility here actually correspond
to poor outcomes. The difference-based relaxation specifies that
the error rate of any given group should not deviate from
the overall error rate by more than the value of :code:`difference_bound`.

.. doctest:: mitigation_reductions
    :options:  +NORMALIZE_WHITESPACE

    >>> from fairlearn.reductions import ErrorRateParity
    >>> from sklearn.metrics import accuracy_score
    >>> X                  = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
    >>> y_true             = np.array([ 1 ,  1 ,  1 ,  1 ,  1,   1 ,  1 ,  0 ,  0 ,  0 ])
    >>> y_pred             = np.array([ 1 ,  1 ,  1 ,  1 ,  0,   0 ,  0 ,  1 ,  0 ,  0 ])
    >>> sensitive_features = np.array(["a", "b", "a", "a", "b", "a", "b", "b", "a", "b"])
    >>> accuracy_summary = MetricFrame(metrics=accuracy_score,
    ...                                y_true=y_true,
    ...                                y_pred=y_pred,
    ...                                sensitive_features=sensitive_features)
    >>> accuracy_summary.overall.item()
    0.6
    >>> accuracy_summary.by_group
    sensitive_feature_0
    a    0.8
    b    0.4
    Name: accuracy_score, dtype: float64
    >>> erp = ErrorRateParity(difference_bound=0.01)
    >>> erp.load_data(X, y_true, sensitive_features=sensitive_features)
    >>> erp.gamma(lambda X: y_pred)
    sign  event  group_id
    +     all    a          -0.2
                 b           0.2
    -     all    a           0.2
                 b          -0.2
    dtype: float64

.. note::

    When providing :code:`ErrorRateParity` to mitigation algorithms, only use
    the constructor. The mitigation algorithm itself then invokes :code:`load_data`.
    The example uses :code:`load_data` to illustrate how :code:`ErrorRateParity`
    instantiates inequalities from :ref:`constraints_binary_classification`.

Alternatively, error rate parity can be relaxed via ratio constraints as

.. math::

   r \leq \dfrac{\P[h(X) \ne Y \given A = a]}{\P[h(X) \ne Y]} \leq \dfrac{1}{r} \quad \forall a

with a :code:`ratio_bound` :math:`r`. The usage is identical with other
constraints:

.. doctest:: mitigation_reductions

    >>> from fairlearn.reductions import ErrorRateParity
    >>> erp = ErrorRateParity(ratio_bound=0.9, ratio_bound_slack=0.01)
    >>> erp.load_data(X, y_true, sensitive_features=sensitive_features)
    >>> erp.gamma(lambda X: y_pred)
    sign  event  group_id
    +     all    a          -0.22
                 b           0.14
    -     all    a           0.16
                 b          -0.24
    dtype: float64


Control features
~~~~~~~~~~~~~~~~

The above examples of :class:`Moment` (:ref:`demographic_parity`,
:ref:`True and False Positive Rate Parity <true_positive_rate_parity>`,
:ref:`equalized_odds` and :ref:`error_rate_parity`) all support the concept
of *control features* when applying their fairness constraints.
A control feature stratifies the dataset, and applies the fairness constraint
within each stratum, but not between strata.
One case this might be useful is a loan scenario, where we might want
to apply a mitigation for the sensitive features while controlling for some
other feature(s).
This should be done with caution, since the control features may have a
correlation with the sensitive features due to historical biases.
In the loan scenario, we might choose to control for income level, on the
grounds that higher income individuals are more likely to be able to repay
a loan.
However, due to historical bias, there is a correlation between the income level
of individuals and their race and gender.


Control features modify the above equations.
Consider a control feature value, drawn from a set of valid values
(that is, :math:`c \in \mathcal{C}`).
The equation given above for Demographic Parity will become:


.. math::
    P[h(X) = 1 | A = a, C = c] = P[h(X) = 1 | C = c] \; \forall a, c

The other constraints acquire similar modifications.


.. _constraints_multi_class_classification:

Fairness constraints for multiclass classification
--------------------------------------------------

Reductions approaches do not support multiclass classification yet at this
point. If this is an important scenario for you please let us know!

.. _constraints_regression:

Fairness constraints for regression
-----------------------------------

The performance objective in the regression scenario is to minimize the
loss of our regressor :math:`h`. The loss can be expressed as
:class:`SquareLoss` or :class:`AbsoluteLoss`. Both take constructor arguments
:code:`min_val` and :code:`max_val` that define the value range within which
the loss is evaluated. Values outside of the value range get clipped.

.. doctest:: mitigation_reductions

    >>> from fairlearn.reductions import SquareLoss, AbsoluteLoss, ZeroOneLoss
    >>> y_true = [0,   0.3, 1,   0.9]
    >>> y_pred = [0.1, 0.2, 0.9, 1.3]
    >>> SquareLoss(0, 2).eval(y_true, y_pred)
    array([0.01, 0.01, 0.01, 0.16])
    >>> # clipping at 1 reduces the error for the fourth entry
    >>> SquareLoss(0, 1).eval(y_true, y_pred)
    array([0.01, 0.01, 0.01, 0.01])
    >>> AbsoluteLoss(0, 2).eval(y_true, y_pred)
    array([0.1, 0.1, 0.1, 0.4])
    >>> AbsoluteLoss(0, 1).eval(y_true, y_pred)
    array([0.1, 0.1, 0.1, 0.1])
    >>> # ZeroOneLoss is identical to AbsoluteLoss(0, 1)
    >>> ZeroOneLoss().eval(y_true, y_pred)
    array([0.1, 0.1, 0.1, 0.1])

When using Fairlearn's reduction techniques for regression it's required to
specify the type of loss by passing the corresponding loss object when
instantiating the object that represents our fairness constraint. The only
supported type of constraint at this point is :class:`BoundedGroupLoss`.

.. _bounded_group_loss:

Bounded Group Loss
~~~~~~~~~~~~~~~~~~

*Bounded group loss* requires the loss of each group to be below a
user-specified amount :math:`\zeta`. If :math:`\zeta` is chosen reasonably
small the losses of all groups are very similar.
Formally, a predictor :math:`h` satisfies bounded group loss at level
:math:`\zeta` under a distribution over :math:`(X, A, Y)` if

.. math::

    \E[loss(Y, h(X)) \given A=a] \leq \zeta \quad \forall a

In the example below we use :class:`BoundedGroupLoss` with
:class:`ZeroOneLoss` on two groups :code:`"a"` and :code:`"b"`.
Group :code:`"a"` has an average loss of :math:`0.05`, while group
:code:`"b"`'s average loss is :math:`0.5`.

.. doctest:: mitigation_reductions
    :options:  +NORMALIZE_WHITESPACE

    >>> from fairlearn.reductions import BoundedGroupLoss, ZeroOneLoss
    >>> from sklearn.metrics import mean_absolute_error
    >>> bgl = BoundedGroupLoss(ZeroOneLoss(), upper_bound=0.1)
    >>> X                  = np.array([[0], [1], [2], [3]])
    >>> y_true             = np.array([0.3, 0.5, 0.1, 1.0])
    >>> y_pred             = np.array([0.3, 0.6, 0.6, 0.5])
    >>> sensitive_features = np.array(["a", "a", "b", "b"])
    >>> mae_frame = MetricFrame(metrics=mean_absolute_error,
    ...                         y_true=y_true,
    ...                         y_pred=y_pred,
    ...                         sensitive_features=pd.Series(sensitive_features, name="SF 0"))
    >>> mae_frame.overall.item()
    0.275
    >>> mae_frame.by_group
    SF 0
    a    0.05
    b    0.50
    Name: mean_absolute_error, dtype: float64
    >>> bgl.load_data(X, y_true, sensitive_features=sensitive_features)
    >>> bgl.gamma(lambda X: y_pred)
    group_id
    a    0.05
    b    0.50
    Name: loss, dtype: float64

.. note::

    In the example above the :code:`BoundedGroupLoss` object does not use the
    :code:`upper_bound` argument. It is only used by reductions techniques
    during the unfairness mitigation. As a result the constraint violation
    detected by :code:`gamma` is identical to the mean absolute error.


Exponentiated Gradient
----------------------

The :class:`ExponentiatedGradient` algorithm in Fairlearn is used to produce models that
satisfy fairness constraints without needing access to sensitive features at deployment time.
This algorithm creates a sequence of re-weighted datasets and retrains the
wrapped classifier on each of these datasets.
To instantiate an :class:`ExponentiatedGradient` model, we need to pass in a base estimator
and fairness constraints. The fairness constraints are typically specified by providing
an upper bound on the difference (or the ratio) between the largest and the smallest
value of some statistic (like a false positive rate) across all groups. This bound is
often referred to as epsilon.

Here is an example of how to instantiate an :class:`ExponentiatedGradient` model:

.. doctest:: mitigation_reductions
    :options:  +NORMALIZE_WHITESPACE

    >>> from fairlearn.datasets import fetch_adult
    >>> from fairlearn.metrics import plot_model_comparison, equal_opportunity_difference
    >>> from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
    >>> from sklearn.ensemble import  RandomForestClassifier
    >>> from sklearn.metrics import accuracy_score
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    >>> from sklearn.compose import ColumnTransformer
    >>> from sklearn.pipeline import Pipeline
    >>> # Fetch and preprocess the data
    >>> X, y = fetch_adult(return_X_y=True, as_frame=True)
    >>> A = X["sex"]
    >>> # Identify features
    >>> categorical_features = X.select_dtypes(include='category').columns.tolist()
    >>> numeric_features = X.select_dtypes(include='number').columns.tolist()
    >>> # Create preprocessor for X
    >>> preprocessor = ColumnTransformer(
    ...     transformers=[
    ...         ('num', 'passthrough', numeric_features),
    ...         ('cat', OneHotEncoder(drop='first'), categorical_features)])
    >>> # Transform y to numerical values
    >>> le = LabelEncoder()
    >>> y = le.fit_transform(y)
    >>> # Create a pipeline with the preprocessor and estimator
    >>> estimator = Pipeline([
    ...     ('preprocessor', preprocessor),
    ...     ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))])
    >>> # Split the data
    >>> X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(X, y, A, test_size=0.2, random_state=42)
    >>> # Train and evaluate the base model
    >>> _ = estimator.fit(X_train, y_train) #variable assignment only there to prevent large output
    >>> y_pred_base = estimator.predict(X_test)
    >>> # Create a list of ExponentiatedGradient models with different epsilons
    >>> epsilons = [0.001, 0.01, 0.05, 0.1, 0.2]
    >>> exp_grad_models = {}
    >>> for eps in epsilons:
    ...     exp_grad_est = ExponentiatedGradient(
    ...         estimator=estimator,
    ...         constraints=EqualizedOdds(difference_bound=eps),
    ...         sample_weight_name="classifier__sample_weight")
    ...     _ = exp_grad_est.fit(X_train, y_train, sensitive_features=A_train) #variable assignment only there to prevent large output
    ...     exp_grad_models[f"ExpGrad (ε={eps})"] = exp_grad_est.predict(X_test)
    >>> # Add the base model predictions
    >>> exp_grad_models["Base Model"] = y_pred_base
    >>> # Plot the comparison
    >>> plot_model_comparison(
    ...     x_axis_metric=accuracy_score,
    ...     y_axis_metric=equal_opportunity_difference,
    ...     y_true=y_test,
    ...     y_preds=exp_grad_models,
    ...     sensitive_features=A_test,
    ...     show_plot=True,
    ...     point_labels =True)
    <Axes: xlabel='accuracy score', ylabel='equal opportunity difference'>

The performance-fairness trade-off learned by the ExponentiatedGradient model is
sensitive to the chosen epsilon value, so epsilon can be treated as a hyperparameter
and iterated over a range of potential values.


.. topic:: References

   .. footbibliography::
