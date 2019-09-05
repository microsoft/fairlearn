# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
import numpy as np
import pandas as pd

from fairlearn.metrics import DemographicParity

import fairlearn.moments as moments


class GridSearchClassification:
    def __init__(self,
                 learner,
                 fairness_metric=DemographicParity(),
                 number_of_lagrange_multipliers=7):
        self.learner = learner
        if not isinstance(fairness_metric, DemographicParity):
            raise RuntimeError("DemographicParity is only currently supported fairness metric")
        self.fairness_metric = fairness_metric
        self.number_of_lagrange_multipliers = number_of_lagrange_multipliers

        self.error_metric = moments.MisclassificationError()
        self.disparity_metric = moments.DP()

    def fit(self, X, Y, protected_attribute, lagrange_multipliers=None):
        # Verify we have a binary classification problem
        unique_labels = np.unique(Y)
        if not set(unique_labels).issubset({0, 1}):
            raise RuntimeError("Supplied Y labels are not 0 or 1")

        # Extract required statistics from protected_attribute
        p0, p1, a0_val = self._generate_protected_attribute_info(protected_attribute)

        # If not supplied, generate array of trial lagrange multipliers
        if lagrange_multipliers is None:
            limit = 1
            if p1 > 0 and p0 / p1 > 1:
                limit = p0 / p1
            lagrange_multipliers = np.linspace(-2 * limit,
                                               2 * limit,
                                               self.number_of_lagrange_multipliers)

        self.all_models = []
        for current_multiplier in lagrange_multipliers:
            # Generate weights array
            sample_weights = self._generate_weights(Y,
                                                    protected_attribute,
                                                    current_multiplier,
                                                    p1 / p0,
                                                    a0_val)

            # Generate Y'
            def f(x): return 1 if x > 0 else 0
            re_labels = np.vectorize(f)(sample_weights)

            # Run the learner
            current_learner = copy.deepcopy(self.learner)
            current_learner.fit(X, re_labels, sample_weight=np.absolute(sample_weights))

            # Append the new model, along with its current_multiplier value
            # to the result
            # Note that we call it a model because it is a learner which has
            # had 'fit' called
            self.all_models.append({"model": current_learner,
                                    "lagrange_multiplier": current_multiplier})

        # Designate a 'best' model
        self.best_model = self._select_best_model(X, Y, protected_attribute, self.all_models)

    def predict(self, X):
        return self.best_model["model"].predict(X)

    def predict_proba(self, X):
        return self.best_model["model"].predict_proba(X)

    def posterior_predict(self, X):
        return [r["model"].predict(X) for r in self.all_models]

    def posterior_predict_proba(self, X):
        return [r["model"].predict_proba(X) for r in self.all_models]

    def _generate_protected_attribute_info(self, protected_attribute):
        unique_labels, counts = np.unique(protected_attribute, return_counts=True)
        if len(unique_labels) > 2:
            raise RuntimeError("Protected Attribute contains more than two unique values")

        p0 = counts[0] / len(protected_attribute)
        p1 = 1 - p0

        return p0, p1, unique_labels[0]

    def _weight_function(self, y_val, a_val, L, p_ratio, a0_val):
        if a_val == a0_val:
            return 2 * y_val - 1 - L * p_ratio
        else:
            return 2 * y_val - 1 + L

    def _generate_weights(self, y, protected_attribute, L, p_ratio, a0_val):
        weight_func = np.vectorize(self._weight_function)
        return weight_func(y, protected_attribute, L, p_ratio, a0_val)

    def _select_best_model(self, X, Y, protected_attribute, model_list):
        self.error_metric.init(X, protected_attribute, pd.Series(Y))
        self.disparity_metric.init(X, protected_attribute, pd.Series(Y))

        for m in model_list:
            current_error_metric = copy.deepcopy(self.error_metric)
            current_disparity_metric = copy.deepcopy(self.disparity_metric)

            def classifier(X): return m["model"].predict(X)
            current_error = current_error_metric.gamma(classifier)[0]
            current_disparity = current_disparity_metric.gamma(classifier).max()

            m["quality"] = - (current_error + current_disparity)

        best_model = max(model_list, key=lambda x: x["quality"])
        return best_model
