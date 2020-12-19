# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Module for visualizing metrics.

This module contains functions for visualizing metrics generated by the
fairlearn.metrics module
"""


from .plot_disparities_in_performance import plot_disparities_in_performance
from .plot_disparities_in_metric import plot_disparities_in_metric

__all__ = [
    "plot_disparities_in_performance",
    "plot_disparities_in_metric"
]
