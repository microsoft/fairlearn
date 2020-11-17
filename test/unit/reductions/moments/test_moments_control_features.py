# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest
import random

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from fairlearn.metrics import MetricFrame
from fairlearn.metrics import selection_rate, true_positive_rate, false_positive_rate
from fairlearn.reductions import DemographicParity, ErrorRateParity,\
    TruePositiveRateParity, FalsePositiveRateParity

# Set up a loan scenario, with three income bands A, B & C and
# one sensitive attribute with values F & G

ibs = ["A", "B", "C"]
sfs = ["F", "G"]

# Numbers for each intersection
n = {
    "A": {"F": 90, "G": 140},
    "B": {"F": 300, "G": 350},
    "C": {"F": 900, "G": 750},
}

# Approval rates for each intersection
f = {
    "A": {"F": 0.6, "G": 0.9},
    "B": {"F": 0.4, "G": 0.7},
    "C": {"F": 0.2, "G": 0.5},
}


def _generate_data():
    IB = []
    SF = []
    PLOAN = []
    Y = []

    for ib in ibs:
        for sf in sfs:
            n_curr = n[ib][sf]
            f_curr = f[ib][sf]

            for i in range(n_curr):
                IB.append(ib)
                SF.append(sf)
                flip = random.random()
                PLOAN.append(flip)
                Y.append(1 if flip < f_curr else 0)

    X = pd.DataFrame(data=np.transpose([IB, SF, PLOAN]), columns=["ctrl", "sens", "ploan"])

    return X, Y


def _simple_compare(moment, metric):
    X, y = _generate_data()
    X_dummy = pd.get_dummies(X)

    est = LogisticRegression()
    est.fit(X_dummy, y)
    y_pred = est.predict(X_dummy)

    target = moment()
    target.load_data(np.asarray(X_dummy), np.asarray(y),
                     sensitive_features=X['sens'],
                     control_features=X['ctrl'])

    results = target.gamma(est.predict)

    mf_pred = MetricFrame(metric, y, y_pred,
                          sensitive_features=X['sens'],
                          control_features=X['ctrl'])
    diffs = mf_pred.by_group - mf_pred.overall

    for ib in ibs:
        for sf in sfs:
            assert diffs[(ib, sf)] == pytest.approx(results[('+', ib, sf)], rel=1e-10, abs=1e-12)
            assert diffs[(ib, sf)] == pytest.approx(-results[('-', ib, sf)], rel=1e-10, abs=1e-12)


def test_demographic_parity():
    _simple_compare(DemographicParity, selection_rate)


def test_error_rate():
    def err_rate(y_true, y_pred):
        return 1 - accuracy_score(y_true, y_pred)

    _simple_compare(ErrorRateParity, err_rate)
