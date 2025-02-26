# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from fairlearn.datasets import fetch_adult
from fairlearn.postprocessing import _relabeling

round_value = 10

dataset = fetch_adult(as_frame=True)
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df[">50k"] = (dataset.target == ">50K") * 1

# Preprocessing
le = preprocessing.LabelEncoder()
df["sex"] = le.fit_transform(df["sex"])
onehot = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "marital-status",
    "occupation",
    "relationship",
    "race",
]
df = pd.get_dummies(df, prefix=onehot, columns=onehot, drop_first=True)

# Reverse the labels to have positive discrimination
tmp = df["sex"].to_list()
for i in range(0, len(tmp)):
    if tmp[i] == 1:
        tmp[i] = 0
    else:
        tmp[i] = 1
df["sex"] = tmp
sensitive = df["sex"]
y = df[">50k"]
X = df.loc[:, ~df.columns.isin(["sex", ">50k", "native-country"])]

y = y.to_numpy()
sensitive = sensitive.to_numpy()
X = X.to_numpy()


def test_relabeling():
    threshold = 0

    clf = DecisionTreeClassifier(max_depth=3, random_state=0)
    clf.fit(X, list(y))
    y_pred = clf.predict(X)
    accuracy = round(accuracy_score(y, y_pred), round_value)
    discrimination = round(_relabeling.discrimination_dataset(y_pred, sensitive),
                           round_value)

    leaves_relabel = _relabeling.leaves_to_relabel(clf, X, y, y_pred, sensitive,
                                                   threshold)

    sum_acc = 0
    sum_disc = 0
    for leaf in leaves_relabel:
        sum_acc += leaf.acc
        sum_disc += leaf.disc
    sum_acc = round(sum_acc, round_value)
    sum_disc = round(sum_disc, round_value)

    _relabeling.relabeling(clf, X, y, y_pred, sensitive, threshold)

    y_pred_relabel = clf.predict(X)
    accuracy_relabel = round(accuracy_score(y, y_pred_relabel), round_value)
    discrimination_relabel = round(
        _relabeling.discrimination_dataset(y_pred_relabel, sensitive), round_value
    )
    new_acc = round(accuracy + sum_acc, round_value)
    new_disc = round(discrimination + sum_disc, round_value)

    # The values are accurate to a certain decimal.
    # It is therefore necessary to accept a small margin of error.
    assert abs(accuracy_relabel - new_acc) <= 0.000000001
    assert abs(discrimination_relabel - new_disc) <= 0.000000001
    assert discrimination_relabel < threshold
