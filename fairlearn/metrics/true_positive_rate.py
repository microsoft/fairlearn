# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np

from . import MetricsResult


def true_positive_rate(y_actual, y_predict, group_id):
    result = MetricsResult()

    groups = np.unique(group_id)

    positives_all = 0
    true_positives_all = 0

    positives = {}
    true_positives = {}
    for group in groups:
        positives[group] = 0
        true_positives[group] = 0

    for y, yhat, a in zip(y_actual, y_predict, group_id):
        if y == 1:
            positives_all += 1
            positives[a] += 1
            if yhat == 1:
                true_positives_all += 1
                true_positives[a] += 1

    result.metric = true_positives_all / positives_all
    for group in groups:
        result.group_metric[group] = true_positives[group] / positives[group]

    return result
