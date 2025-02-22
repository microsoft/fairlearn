# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Utilities for plotting curves."""

import pandas as pd
from sklearn.utils.validation import check_is_fitted

from ._constants import _MATPLOTLIB_IMPORT_ERROR_MESSAGE
from ._threshold_optimizer import ThresholdOptimizer

_debug_colors = None
_debug_ncolors = 10
_debug_colormap = {}


def _get_debug_color(key):
    global _debug_colors, _debug_ncolors, _debug_colormap
    try:
        import matplotlib.cm as cm
        import matplotlib.colors
    except ImportError:
        raise RuntimeError(_MATPLOTLIB_IMPORT_ERROR_MESSAGE)
    if _debug_colors is None:
        tab_norm = matplotlib.colors.Normalize(vmin=0, vmax=7)
        tab_scalarMap = cm.ScalarMappable(norm=tab_norm, cmap="Dark2")
        _debug_colors = [tab_scalarMap.to_rgba(x) for x in range(_debug_ncolors)]

    if key not in _debug_colormap:
        color = _debug_colors[len(_debug_colormap) % _debug_ncolors]
        _debug_colormap[key] = color
    return _debug_colormap[key]


def _plot_solution(ax, x_best, y_best, solution_label, xlabel, ylabel):
    """Plot the given solution with appropriate labels."""
    if y_best is None:
        ax.axvline(x=x_best, label=solution_label, ls="--")
    else:
        ax.plot(x_best, y_best, "m*", ms=10, label=solution_label)
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _plot_overall_tradeoff_curve(ax, overall_tradeoff_curve: pd.DataFrame) -> None:
    """Plot the overall tradeoff curve."""
    ax.plot(
        overall_tradeoff_curve["x"],
        overall_tradeoff_curve["y"],
        c="b",
        ls="--",
        lw=2.0,
        label="overall tradeoff curve",
    )


def _plot_overlap(ax, x_grid, y_min):
    """Plot the overlap region."""
    highlight_color = [0.95, 0.90, 0.40]
    (line,) = ax.plot(x_grid, y_min, color=highlight_color, lw=8, label="overlap")
    line.zorder -= 1


def _plot_curve(ax, sensitive_feature, x_col, y_col, points):
    """Plot the given curve with labels."""
    color = _get_debug_color(sensitive_feature)
    ax.plot(
        points[x_col],
        points[y_col],
        c=color,
        ls="-",
        lw=2.0,
        label="sensitive feature = " + str(sensitive_feature),
    )


def _raise_if_not_threshold_optimizer(obj):
    if not isinstance(obj, ThresholdOptimizer):
        raise ValueError(
            "Argument {} needs to be of type {}.".format(obj.__name__, ThresholdOptimizer.__name__)
        )


def plot_threshold_optimizer(threshold_optimizer: ThresholdOptimizer, ax=None, show_plot=True):
    r"""Plot the chosen solution of the threshold optimizer.

    For :class:`.ThresholdOptimizer` objects that have their
    constraint set to :code:`demographic_parity` this will result in a
    selection/error curve plot. For :class:`.ThresholdOptimizer`
    objects that have their constraint set to :code:`equalized_odds` this will
    result in a ROC curve plot.

    Parameters
    ----------
    threshold_optimizer : :class:`.ThresholdOptimizer`
        The `ThresholdOptimizer` instance for which the results should be
        illustrated.
    ax : :class:`matplotlib.axes.Axes`, default = None
        A custom `matplotlib.axes.Axes` object to use for the plots.
    show_plot : bool, default = True
        Whether or not the generated plot should be shown, default True
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError(_MATPLOTLIB_IMPORT_ERROR_MESSAGE)

    _raise_if_not_threshold_optimizer(threshold_optimizer)
    check_is_fitted(threshold_optimizer)

    if ax is None:
        ax = plt.axes()

    for sensitive_feature_value in threshold_optimizer._tradeoff_curve.keys():
        _plot_curve(
            ax,
            sensitive_feature_value,
            "x",
            "y",
            threshold_optimizer._tradeoff_curve[sensitive_feature_value],
        )

    if threshold_optimizer.constraints == "equalized_odds":
        _plot_overlap(ax, threshold_optimizer._x_grid, threshold_optimizer._y_min)
        _plot_solution(
            ax,
            threshold_optimizer._x_best,
            threshold_optimizer._y_best,
            "solution",
            "$P[\\hat{Y}=1|Y=0]$",
            "$P[\\hat{Y}=1|Y=1]$",
        )
    else:
        if threshold_optimizer.tol:
            _plot_solution_with_tol(threshold_optimizer, ax)

        else:
            _plot_overall_tradeoff_curve(ax, threshold_optimizer._overall_tradeoff_curve)
            _plot_solution(
                ax,
                threshold_optimizer._x_best,
                None,
                "solution",
                threshold_optimizer.x_metric_,
                threshold_optimizer.y_metric_,
            )

    if show_plot:
        plt.show()


def _plot_solution_with_tol(threshold_optimizer: ThresholdOptimizer, ax):
    for sensitive_feature_value, x_best in threshold_optimizer._x_best_per_group.items():
        color = _get_debug_color(sensitive_feature_value)
        ax.axvline(
            x=x_best,
            label=f"constraint (sensitive feature = {sensitive_feature_value})",
            ls="--",
            c=color,
        )
        ax.legend()
        ax.set_xlabel(threshold_optimizer.x_metric_)
        ax.set_ylabel(threshold_optimizer.y_metric_)
