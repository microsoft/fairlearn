.. _preprocessing:

Preprocessing
=============

.. currentmodule:: fairlearn.preprocessing

Preprocessing algorithms transform the dataset to mitigate possible unfairness
present in the data.
Preprocessing algorithms in Fairlearn follow the :class:`sklearn.base.TransformerMixin`
class, meaning that they can :code:`fit` to the dataset and :code:`transform` it
(or :code:`fit_transform` to fit and transform in one go).

.. _correlation_remover:

Correlation Remover
-------------------

Sensitive features can be correlated with non-sensitive features in the dataset.
By applying the :code:`CorrelationRemover`, these correlations are projected away
while details from the original data are retained as much as possible (as measured
by the least-squares error). The user can control the level of projection via the
:code:`alpha` parameter.

In mathematical terms, assume we have the original dataset :math:`\mathbf{X}`, which
contains a set of **sensitive features** denoted by :math:`\mathbf{S}` and a set of
**non-sensitive features** denoted by :math:`\mathbf{Z}`. The goal is to remove
correlations between the sensitive features and the non-sensitive features.

Let :math:`m_s` and :math:`m_{ns}` denote the number of sensitive and non-sensitive
features, respectively.
Let :math:`\bar{\mathbf{s}}` represent the mean of the sensitive features, *i.e.*,
:math:`\bar{\mathbf{s}} = [\bar{s}_1, \dots, \bar{s}_{m_s}]^\top`, where
:math:`\bar{s}_j` is the mean of the :math:`j\text{-th}` sensitive feature.

For each non-sensitive feature :math:`\mathbf{z}_j\in\mathbb{R}^n`, where :math:`j=1,\dotsc,m_{ns}`,
we compute an optimal weight vector :math:`\mathbf{w}_j^* \in \mathbb{R}^{m_s}` that minimizes the
following least squares objective:

.. math::

    \min _{\mathbf{w}} \| \mathbf{z}_j - (\mathbf{S}-\mathbf{1}_n\times\bar{\mathbf{s}}^\top) \mathbf{w} \|_2^2

where :math:`\mathbf{1}_n` is the all-one vector in :math:`\mathbb{R}^n`.

In other words, :math:`\mathbf{w}_j^*` is the solution to a linear regression problem where we
project :math:`\mathbf{z}_j` onto the centered sensitive features. The weight matrix
:math:`\mathbf{W}^* = (\mathbf{w}_1^*, \dots, \mathbf{w}_{m_{ns}}^*)` is thus obtained by solving
this regression for each non-sensitive feature.

Once we have the optimal weight matrix :math:`\mathbf{W}^*`, we compute the **residual
non-sensitive features** :math:`\mathbf{Z}^*` as follows:

.. math::

    \mathbf{Z}^* = \mathbf{Z} - (\mathbf{S}-\mathbf{1}_n\times\bar{\mathbf{s}}^\top) \mathbf{W}^*

The columns in :math:`\mathbf{S}` will be dropped from the dataset :math:`\mathbf{X}`, and
:math:`\mathbf{Z}^*` will replace the original non-sensitive features :math:`\mathbf{Z}`, but the
hyper parameter :math:`\alpha` does allow you to tweak the amount of filtering that gets applied:

.. math::
    \mathbf{X}_{\text{tfm}} = \alpha \mathbf{X}_{\text{filtered}} + (1-\alpha) \mathbf{X}_{\text{orig}}

Note that the lack of correlation does not imply anything about statistical dependence.
In particular, since correlation measures linear relationships, it might still be
possible that non-linear relationships exist in the data. Therefore, we expect this
to be most appropriate as a preprocessing step for (generalized) linear models.

In the example below, the `Diabetes 130-Hospitals <https://www.openml.org/d/43874>`_
is loaded and the correlation between the African American race and
the non-sensitive features is removed. This dataset contains more races,
but in example we will only focus on the African American race.
The :code:`CorrelationRemover` will drop the sensitive features from the dataset.

.. doctest:: mitigation_preprocessing
    :options:  +NORMALIZE_WHITESPACE

    >>> from fairlearn.preprocessing import CorrelationRemover
    >>> import pandas as pd
    >>> from fairlearn.datasets import fetch_diabetes_hospital
    >>> data = fetch_diabetes_hospital()
    >>> X = data.data[["race", "time_in_hospital", "had_inpatient_days", "medicare"]]
    >>> X = pd.get_dummies(X)
    >>> X = X.drop(["race_Asian",
    ...                     "race_Caucasian",
    ...                     "race_Hispanic",
    ...                     "race_Other",
    ...                     "race_Unknown",
    ...                     "had_inpatient_days_False",
    ...                     "medicare_False"], axis=1)
    >>> cr = CorrelationRemover(sensitive_feature_ids=['race_AfricanAmerican'])
    >>> cr.fit(X)
    CorrelationRemover(sensitive_feature_ids=['race_AfricanAmerican'])
    >>> X_transform = cr.transform(X)

In the visualization below, we see the correlation values in the
original dataset. We are particularly interested in the correlations
between the 'race_AfricanAmerican' column and the three non-sensitive features
'time_in_hospital', 'had_inpatient_days' and 'medicare_True'. The target
variable is also included in these visualization for completeness, and it is
defined as a binary feature which indicated whether the readmission of a patient
occurred within 30 days of the release. We see that 'race_AfricanAmerican' is
not highly correlated with the three mentioned features, but we want to remove
these correlations nonetheless. The code for generating the correlation matrix
can be found in
`this example notebook
<../../auto_examples/plot_correlationremover_before_after.html>`_.

.. figure:: ../../auto_examples/images/sphx_glr_plot_correlationremover_before_after_001.png
    :align: center
    :target: ../../auto_examples/plot_correlationremover_before_after.html

In order to see the effect of :class:`CorrelationRemover`, we visualize
how the correlation matrix has changed after the transformation of the
dataset. Due to rounding, some of the 0.0 values appear as -0.0. Either
way, the :code:`CorrelationRemover` successfully removed all correlation
between 'race_AfricanAmerican' and the other columns while retaining
the correlation between the other features.

.. figure:: ../../auto_examples/images/sphx_glr_plot_correlationremover_before_after_002.png
    :align: center
    :target: ../../auto_examples/plot_correlationremover_before_after.html

We can also use the :code:`alpha` parameter with for instance :math:`\alpha=0.5`
to control the level of filtering between the sensitive and non-sensitive features.

.. doctest:: mitigation_preprocessing

    >>> cr = CorrelationRemover(sensitive_feature_ids=['race_AfricanAmerican'], alpha=0.5)
    >>> cr.fit(X)
    CorrelationRemover(alpha=0.5, sensitive_feature_ids=['race_AfricanAmerican'])
    >>> X_transform = cr.transform(X)

As we can see in the visualization below, not all correlation between
'race_AfricanAmerican' and the other columns was removed. This is exactly what
we would expect with :math:`\alpha=0.5`.

.. figure:: ../../auto_examples/images/sphx_glr_plot_correlationremover_before_after_003.png
    :align: center
    :target: ../../auto_examples/plot_correlationremover_before_after.html


Reweighing
----------
Another approach to remove discrimination from the data set is to weight each data entry
such that there is a discrimination free balance, as presented in Section 5.2 of
:footcite:`kamiran2012`. The algorithm reduces the discrimination to 0
while maintaining the overall positive class probability.

Given a dataset :math:`D`, sensitive features :math:`S`, and classification targets :math:`Y`.
If the dataset :math:`D` is unbiased, the class probabilities are statistically
independent of the sensitive features, so for any sensitive feature :math:`s \in S` and target :math:`y \in Y`:

.. math::
    P_{\text{indep}}[X(S)=s \land X(Y) = y] = \frac{|\{X \in D \mid X(S) = s\}|}{|D|} \times \frac{|\{X \in D \mid X(Y) = y\}|}{|D|}.

However, a dataset's observed class probabilities,

.. math::
    P_{\text{obs}}[X(S)=s ~\land~ X(Y) = y] = \frac{|\{X \in D \mid X(S) = s ~\land~ X(Y) = y\}|}{|D|},

are often different of :math:`P_{\text{ind}}`, leading to under or overrepresentation of specific
sensitive groups. To compensate for this bias, the algorithm as presented in Section 5.2 of
:footcite:`kamiran2012` applies a weight :math:`W(X)` to each data point :math:`X \in D` of:

.. math::
    W(X) = \frac{P_{\text{indep}}[X(S)=S \land X(Y) = Y]}{P_{\text{obs}}[X(S)=S ~\land~ X(Y) = Y]}

In the following example, we firstly train a simple Bernoulli estimator, and set a
baseline for the accuracy of this estimator. In particular, we measure the
disparity between the sexes.

.. doctest:: without_reweighing
    :options:  +NORMALIZE_WHITESPACE

    >>> from fairlearn.metrics import MetricFrame
    >>> from fairlearn.preprocessing import Reweighing
    >>> from sklearn.datasets import fetch_openml
    >>> from sklearn.metrics import accuracy_score
    >>> from sklearn.naive_bayes import BernoulliNB
    >>> import pandas as pd
    >>> data = fetch_openml(data_id=1590, as_frame=True)
    >>> X = pd.get_dummies(data.data)
    >>> Y = (data.target == ">50K") * 1
    >>> A = (data.data["sex"] == "Male") * 1
    >>> est = BernoulliNB()
    >>> est.fit(X, Y);
    >>> pred_original = est.predict(X)

    >>> diff_original = MetricFrame(
    ...     metrics=accuracy_score,
    ...     y_true=Y,
    ...     y_pred=pred_original,
    ...     sensitive_features=A,
    ... ).difference()
    >>> print("%.3f" % diff_original)
    0.231

This is quite a significant difference. Now, by applying the
reweighing preprocessing algorithm, we can see this disparity
decreases.

.. doctest:: with_reweighing
    :options:  +NORMALIZE_WHITESPACE

    >>> from fairlearn.metrics import MetricFrame
    >>> from fairlearn.preprocessing import Reweighing
    >>> from sklearn.datasets import fetch_openml
    >>> from sklearn.metrics import accuracy_score
    >>> from sklearn.naive_bayes import BernoulliNB
    >>> import pandas as pd
    >>> data = fetch_openml(data_id=1590, as_frame=True)
    >>> X = pd.get_dummies(data.data)
    >>> Y = (data.target == ">50K") * 1
    >>> A = (data.data["sex"] == "Male") * 1
    >>> est = BernoulliNB()
    >>> metaest = Reweighing(est, passthrough_sensitive_features=False)
    >>> metaest.fit(X, Y, sensitive_features=A);
    >>> pred_mitigated = est.predict(X)

    >>> diff_mitigated = MetricFrame(
    ...     metrics=accuracy_score,
    ...     y_true=Y,
    ...     y_pred=pred_mitigated,
    ...     sensitive_features=A,
    ... ).difference()
    >>> print("%.3f" % diff_mitigated)
    0.142

References
----------

.. footbibliography::
