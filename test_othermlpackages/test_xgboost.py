# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import pytest

from fairlearn.reductions import DemographicParity

from . import package_test_common as ptc

xgb = pytest.importorskip("xgboost")


def test_expgrad_classification():
    estimator = xgb.XGBClassifier()
    disparity_moment = DemographicParity()

    ptc.run_expgrad_classification(estimator, disparity_moment)


def test_gridsearch_classification():
    estimator = xgb.XGBClassifier()
    disparity_moment = DemographicParity()

    ptc.run_gridsearch_classification(estimator, disparity_moment)


def test_thresholdoptimizer_classification():
    estimator = xgb.XGBClassifier()

    ptc.run_thresholdoptimizer_classification(estimator)
