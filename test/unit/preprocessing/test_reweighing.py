# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.
from fairlearn.metrics import MetricFrame
from fairlearn.preprocessing import Reweighing
from fairlearn.preprocessing._reweighing import calculate_weights
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB, CategoricalNB
import numpy as np
import pandas as pd


def main_test(est, X, Y, A, pass_sf=False, assert_improvement=False):
    est.fit(X, Y)
    pred_original = est.predict(X)
    diff_original = MetricFrame(
        metrics=accuracy_score,
        y_true=Y,
        y_pred=pred_original,
        sensitive_features=A,
    ).difference()

    metaest = Reweighing(est, passthrough_sensitive_features=pass_sf)
    metaest.fit(X, Y, sensitive_features=A)

    pred_mitigated = est.predict(X)
    diff_mitigated = MetricFrame(
        metrics=accuracy_score,
        y_true=Y,
        y_pred=pred_mitigated,
        sensitive_features=A,
    ).difference()

    if assert_improvement:
        assert diff_mitigated < diff_original


def test_single_sensitive_feature():
    data = fetch_openml(data_id=1590, as_frame=True)
    X = pd.get_dummies(data.data)
    Y = (data.target == ">50K") * 1
    A = (data.data["sex"] == "Male") * 1

    est = BernoulliNB()
    main_test(est, X, Y, A, assert_improvement=True)


def test_multiple_sensitive_features():
    data = fetch_openml(data_id=1590, as_frame=True)
    X = pd.get_dummies(data.data)
    Y = (data.target == ">50K") * 1
    A = np.array(
        [(data.data["sex"] == "Male") * 1, (data.data["race"] == "Black") * 1]
    ).T
    est = BernoulliNB()
    main_test(est, X, Y, A, assert_improvement=True)


def test_categorical():
    data = fetch_openml(data_id=469, as_frame=True)
    X = pd.get_dummies(data.data)
    Y = data.target
    A = data.data["Ethnic"]

    est = CategoricalNB()
    main_test(est, X, Y, A)


def test_calculation():
    # for all: w = 2 x 2 / 4 x 1 = 1.0
    assert calculate_weights([0, 0, 1, 1], [0, 1, 0, 1]).tolist() == [1.0] * 4
    # for 0,0: w = 4 x 4 / (6 x 3)
    # for 0,1 and 1,0: w = 4 x 2 / (6 x 1)
    # for 1,1: w = 2 x 2 / (6 x 1)
    assert calculate_weights([0, 0, 1, 1, 0, 0], [0, 1, 0, 1, 0, 0]).tolist() == [
        16 / 18,
        8 / 6,
        8 / 6,
        4 / 6,
        16 / 18,
        16 / 18,
    ]
