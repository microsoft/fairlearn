.. _api_reference:

API Docs
========

:py:mod:`fairlearn.datasets`
============================

.. automodule:: fairlearn.datasets
    :no-members:
    :no-inherited-members:

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: function.rst

   fetch_acs_income
   fetch_adult
   fetch_bank_marketing
   fetch_boston
   fetch_credit_card
   fetch_diabetes_hospital

:py:mod:`fairlearn.metrics`
===========================

.. automodule:: fairlearn.metrics
    :no-members:
    :no-inherited-members:

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: function.rst

    count
    demographic_parity_difference
    demographic_parity_ratio
    equalized_odds_difference
    equalized_odds_ratio
    equal_opportunity_difference
    equal_opportunity_ratio
    false_negative_rate
    false_positive_rate
    make_derived_metric
    mean_prediction
    plot_model_comparison
    selection_rate
    true_negative_rate
    true_positive_rate


.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

    MetricFrame

:py:mod:`fairlearn.postprocessing`
==================================

.. automodule:: fairlearn.postprocessing
    :no-members:
    :no-inherited-members:

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   ThresholdOptimizer

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: function.rst

   plot_threshold_optimizer

:py:mod:`fairlearn.preprocessing`
=================================

.. automodule:: fairlearn.preprocessing
    :no-members:
    :no-inherited-members:

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   CorrelationRemover
   PrototypedRepresenter

:py:mod:`fairlearn.reductions`
==============================

.. automodule:: fairlearn.reductions
    :no-members:
    :no-inherited-members:

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   AbsoluteLoss
   BoundedGroupLoss
   ClassificationMoment
   DemographicParity
   EqualizedOdds
   ErrorRate
   ErrorRateParity
   ExponentiatedGradient
   TruePositiveRateParity
   FalsePositiveRateParity
   UtilityParity
   GridSearch
   LossMoment
   Moment
   BoundedGroupLoss
   SquareLoss
   ZeroOneLoss

:py:mod:`fairlearn.adversarial`
===============================

.. automodule:: fairlearn.adversarial
    :no-members:
    :no-inherited-members:

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   AdversarialFairnessClassifier
   AdversarialFairnessRegressor

:py:mod:`fairlearn.experimental`
================================

.. automodule:: fairlearn.experimental
    :no-members:
    :no-inherited-members:

.. currentmodule:: fairlearn

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: function.rst

   metrics._plotter.plot_metric_frame
