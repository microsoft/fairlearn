# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from fairlearn.reductions._moments import ClassificationMoment
from fairlearn.reductions._moments.moment import Moment

from ._constants import (
    _ACCURACY_MUL,
    _INDENTATION,
    _MIN_ITER,
    _PRECISION,
    _REGRET_CHECK_INCREASE_T,
    _REGRET_CHECK_START_T,
    _SHRINK_ETA,
    _SHRINK_REGRET,
)
from ._lagrangian import _Lagrangian

logger = logging.getLogger(__name__)


class ExponentiatedGradient(BaseEstimator, MetaEstimatorMixin):
    """An Estimator which implements the exponentiated gradient reduction.

    The exponentiated gradient algorithm is described in detail by
    :footcite:t:`agarwal2018reductions`.

    Read more in :ref:`sphx_glr_auto_examples_plot_mitigation_pipeline.py`.

    .. versionchanged:: 0.3.0
        Was a function before, not a class

    .. versionchanged:: 0.4.6
        Requires 0-1 labels for classification problems

    Parameters
    ----------
    estimator : estimator
        An estimator implementing methods :code:`fit(X, y, sample_weight)` and
        :code:`predict(X)`, where `X` is the matrix of features, `y` is the
        vector of labels (binary classification) or continuous values
        (regression), and `sample_weight` is a vector of weights.
        In binary classification labels `y` and predictions returned by
        :code:`predict(X)` are either 0 or 1.
        In regression values `y` and predictions are continuous.
    constraints : fairlearn.reductions.Moment
        The fairness constraints expressed as a :class:`~Moment`.
    objective : fairlearn.reductions.Moment | None
        The objective expressed as a :class:`~Moment`. The default is
        :code:`ErrorRate()` for binary classification and
        :code:`MeanLoss(...)` for regression.
    eps : float
        Allowed fairness constraint violation; the solution is guaranteed to
        have the error within :code:`2*best_gap` of the best error under
        constraint `eps`; the constraint violation is at most
        :code:`2*(eps+best_gap)`.

        .. versionchanged:: 0.5.0
            :code:`eps` is now only responsible for setting the L1 norm bound
            in the optimization
    max_iter : int
        Maximum number of iterations

        .. versionadded:: 0.5.0
            Used to be :code:`T`

    nu : float | None
        Convergence threshold for the duality gap, corresponding to a
        conservative automatic setting based on the statistical uncertainty
        in measuring classification error

    eta0 : float
        Initial setting of the learning rate

        .. versionadded:: 0.5.0
            Used to be :code:`eta_mul`

    run_linprog_step : bool
        if True each step of exponentiated gradient is followed by the saddle
        point optimization over the convex hull of classifiers returned so
        far; default True

        .. versionadded:: 0.5.0

    sample_weight_name : str
        Name of the argument to `estimator.fit()` which supplies the sample weights
        (defaults to `sample_weight`)

        .. versionadded:: 0.5.0

    References
    ----------
    .. footbibliography::

    """

    def __init__(
        self,
        estimator,
        constraints: Moment,
        *,
        objective: Moment | None = None,
        eps: float = 0.01,
        max_iter: int = 50,
        nu: float | None = None,
        eta0: float = 2.0,
        run_linprog_step: bool = True,
        sample_weight_name: str = "sample_weight",
    ):  # noqa: D103
        self.estimator = estimator
        self.constraints = constraints
        self.objective = objective
        self.eps = eps
        self.max_iter = max_iter
        self.nu = nu
        self.eta0 = eta0
        self.run_linprog_step = run_linprog_step
        self.sample_weight_name = sample_weight_name

    def fit(self, X, y, **kwargs):
        """Return a fair classifier under specified fairness constraints.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            Feature data
        y : numpy.ndarray, pandas.DataFrame, pandas.Series, or list
            Label vector
        """
        self.lambda_vecs_EG_ = pd.DataFrame()
        self.lambda_vecs_LP_ = pd.DataFrame()

        logger.debug("...Exponentiated Gradient STARTING")

        B = 1 / self.eps
        lagrangian = _Lagrangian(
            X=X,
            y=y,
            estimator=self.estimator,
            constraints=self.constraints,
            B=B,
            objective=self.objective,
            sample_weight_name=self.sample_weight_name,
            **kwargs,
        )

        theta = pd.Series(0, lagrangian.constraints.index)

        # Stores the weights of the best-response estimators without LP and before averaging
        Qsum = pd.Series(dtype="float64")

        # Stores the duality gap of the Exponentiated Gradient (EG) at each iteration
        gaps_EG: list[float] = []

        # Stores the best gap at each iteration among the ones EG and Linear Programming (LP) yield
        gaps: list[float] = []

        # Stores the best-reponse estimators' weights at each iteration
        # The appended weights come either from (EG) or (LP), depending on their respective gaps
        Qs: list[pd.Series] = []

        last_regret_checked = _REGRET_CHECK_START_T
        last_gap = np.inf
        for t in range(0, self.max_iter):
            logger.debug("...iter=%03d", t)

            # set lambdas for every constraint
            lambda_vec = B * np.exp(theta) / (1 + np.exp(theta).sum())
            self.lambda_vecs_EG_[t] = lambda_vec
            lambda_EG = self.lambda_vecs_EG_.mean(axis=1)

            # select classifier according to best_h method
            h, h_idx = lagrangian.best_h(lambda_vec)

            if t == 0:
                if self.nu is None:
                    self.nu = (
                        _ACCURACY_MUL
                        * (h(X) - self.constraints._y_as_series).abs().std()
                        / np.sqrt(self.constraints.total_samples)
                    )
                eta = self.eta0 / B
                logger.debug(
                    "...eps=%.3f, B=%.1f, nu=%.6f, max_iter=%d",
                    self.eps,
                    B,
                    self.nu,
                    self.max_iter,
                )

            if h_idx not in Qsum.index:
                Qsum.at[h_idx] = 0.0
            Qsum[h_idx] += 1.0
            gamma = lagrangian.gammas[h_idx]
            Q_EG = Qsum / Qsum.sum()
            result_EG = lagrangian.eval_gap(Q_EG, lambda_EG, self.nu)
            gap_EG = result_EG.gap()
            gaps_EG.append(gap_EG)

            if t == 0 or not self.run_linprog_step:
                gap_LP = np.inf
            else:
                # saddle point optimization over the convex hull of
                # classifiers returned so far
                Q_LP, self.lambda_vecs_LP_[t], result_LP = lagrangian.solve_linprog(self.nu)
                gap_LP = result_LP.gap()

            # keep values from exponentiated gradient or linear programming
            if gap_EG < gap_LP:
                Qs.append(Q_EG)
                gaps.append(gap_EG)
            else:
                Qs.append(Q_LP)
                gaps.append(gap_LP)

            logger.debug(
                "%seta=%.6f, L_low=%.3f, L=%.3f, L_high=%.3f, gap=%.6f, disp=%.3f, "
                "err=%.3f, gap_LP=%.6f",
                _INDENTATION,
                eta,
                result_EG.L_low,
                result_EG.L,
                result_EG.L_high,
                gap_EG,
                result_EG.gamma.max(),
                result_EG.error,
                gap_LP,
            )

            if (gaps[t] < self.nu) and (t >= _MIN_ITER):
                # solution found
                break

            # update regret
            if t >= last_regret_checked * _REGRET_CHECK_INCREASE_T:
                best_gap = min(gaps_EG)

                if best_gap > last_gap * _SHRINK_REGRET:
                    eta *= _SHRINK_ETA
                last_regret_checked = t
                last_gap = best_gap

            # update theta based on learning rate
            theta += eta * (gamma - self.constraints.bound())

        # retain relevant result data
        gaps_series = pd.Series(gaps)
        gaps_best = gaps_series[gaps_series <= gaps_series.min() + _PRECISION]
        self.best_iter_ = gaps_best.index[-1]
        self.best_gap_ = gaps[self.best_iter_]
        self.weights_ = Qs[self.best_iter_]
        self._hs = lagrangian.hs
        for h_idx in self._hs.index:
            if h_idx not in self.weights_.index:
                self.weights_.at[h_idx] = 0.0

        self.last_iter_ = len(Qs) - 1
        self.predictors_ = lagrangian.predictors
        self.n_oracle_calls_ = lagrangian.n_oracle_calls
        self.n_oracle_calls_dummy_returned_ = lagrangian.n_oracle_calls_dummy_returned
        self.oracle_execution_times_ = lagrangian.oracle_execution_times
        self.lambda_vecs_ = lagrangian.lambdas

        logger.debug(
            "...eps=%.3f, B=%.1f, nu=%.6f, max_iter=%d",
            self.eps,
            B,
            self.nu,
            self.max_iter,
        )
        logger.debug(
            "...last_iter=%d, best_iter=%d, best_gap=%.6f, n_oracle_calls=%d, n_hs=%d",
            self.last_iter_,
            self.best_iter_,
            self.best_gap_,
            lagrangian.n_oracle_calls,
            len(lagrangian.predictors),
        )
        return self

    def predict(self, X, random_state=None):
        """Provide predictions for the given input data.

        Predictions are randomized, i.e., repeatedly calling `predict` with
        the same feature data may yield different output. This
        non-deterministic behavior is intended and stems from the nature of
        the exponentiated gradient algorithm.

        Notes
        -----
        A fitted ExponentiatedGradient has an attribute `predictors_`, an
        array of predictors, and an attribute `weights_`, an array of
        non-negative floats of the same length.
        The prediction on each data point in `X` is obtained by first picking
        a random predictor according to the probabilities in `weights_` and
        then applying it. Different predictors can be chosen on different data
        points.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            Feature data
        random_state : int or RandomState instance, default=None
            Controls random numbers used for randomized predictions. Pass an
            int for reproducible output across multiple function calls.

        Returns
        -------
        Scalar or vector
            The prediction. If `X` represents the data for a single example
            the result will be a scalar. Otherwise the result will be a vector
        """
        check_is_fitted(self)
        random_state = check_random_state(random_state)

        if isinstance(self.constraints, ClassificationMoment):
            positive_probs = self._pmf_predict(X)[:, 1]
            return (positive_probs >= random_state.rand(len(positive_probs))) * 1
        else:
            pred = self._pmf_predict(X)
            randomized_pred = np.zeros(pred.shape[0])
            for i in range(pred.shape[0]):
                randomized_pred[i] = random_state.choice(pred.iloc[i, :], p=self.weights_)
            return randomized_pred

    def _pmf_predict(self, X):
        """Probability mass function for the given input data.

        For each data point, provide the probabilities with which 0 and 1 is
        returned as a prediction.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            Feature data

        Returns
        -------
        pandas.DataFrame
            Array of tuples with the probabilities of predicting 0 and 1.
        """
        check_is_fitted(self)

        pred = pd.DataFrame()
        for t in range(len(self._hs)):
            if self.weights_[t] == 0:
                pred[t] = np.zeros(len(X))
            else:
                pred[t] = self._hs[t](X)

        if isinstance(self.constraints, ClassificationMoment):
            positive_probs = pred[self.weights_.index].dot(self.weights_).to_frame()
            return np.concatenate((1 - positive_probs, positive_probs), axis=1)
        else:
            return pred
