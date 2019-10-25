# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""This module implements the Lagrangian reduction of fair binary
classification to standard binary classification.
"""

import logging
import numpy as np
import pandas as pd
from fairlearn.reductions import Reduction
from ._constants import _ACCURACY_MUL, _REGRET_CHECK_START_T, _REGRET_CHECK_INCREASE_T, \
    _SHRINK_REGRET, _SHRINK_ETA, _MIN_T, _RUN_LP_STEP, _PRECISION, _INDENTATION
from ._lagrangian import _Lagrangian
from fairlearn import _KW_SENSITIVE_FEATURES

logger = logging.getLogger(__name__)


def _mean_pred(X, hs, weights):
    """Return a weighted average of predictions produced by classifiers in hs"""
    pred = pd.DataFrame()
    for t in range(len(hs)):
        pred[t] = hs[t](X)
    return pred[weights.index].dot(weights)


class ExponentiatedGradientResult:
    """Class to hold the result of an ExponentiatedGradient
    estimator
    """
    def __init__(self, best_classifier, best_gap, classifiers, weights, last_t, best_t,
                 n_oracle_calls):
        """ Result object for the exponentiated gradient reduction operation.
        """

        """ A function that maps a DataFrame X containing covariates to a Series containing the
        corresponding probabilistic decisions in [0,1]
        """
        self._best_classifier = best_classifier

        """ The quality of best_classifier; if the algorithm has converged then best_gap <= nu;
        the solution best_classifier is guaranteed to have the classification error within
        2*best_gap of the best error under constraint eps; the constraint violation is at most
        2*(eps+best_gap)
        """
        self._best_gap = best_gap

        """ The base classifiers generated (instances of estimator).
        """
        self._classifiers = classifiers

        """ The weights of those classifiers within best_classifier.
        """
        self._weights = weights

        """ The last executed iteration; always last_t < T.
        """
        self._last_t = last_t

        """ The iteration in which best_classifier was obtained.
        """
        self._best_t = best_t

        """ The number of times the estimator was called.
        """
        self._n_oracle_calls = n_oracle_calls

    def _as_dict(self):
        return {
            "best_classifier": self._best_classifier,
            "best_gap": self._best_gap,
            "classifiers": self._classifiers,
            "weights": self._weights,
            "last_t": self._last_t,
            "best_t": self._best_t,
            "n_oracle_calls": self._n_oracle_calls
        }


class ExponentiatedGradient(Reduction):
    """An Estimator which implements the exponentiated gradient approach to
    reductions described by `Agarwal et al. (2018) <https://arxiv.org/abs/1803.02453>`_.
    """
    def __init__(self, estimator, constraints, eps=0.01, T=50, nu=None, eta_mul=2.0):
        """ The constructor for a mitigator object that applies the exponentiated gradient
        reduction to a provided estimator to achieve the given constraints.

        :param estimator: an estimator implementing methods fit(X, y, sample_weight) and
            predict(X), where X is the set of features, y is the set of labels, and
            sample_weight is a set of weights; labels y and predictions returned by predict(X)
            are either 0 or 1.
        :type estimator: an estimator
        :param constraints: the disparity constraints expressed as moments
        :type constraints: fairlearn.reductions.moments.Moment
        :param eps: allowed fairness constraint violation (default 0.01)
        :type eps: float
        :param T: max number of iterations (default 50)
        :type T: int
        :param nu: convergence threshold for the duality gap (default None), corresponding to a
            conservative automatic setting based on the statistical uncertainty in measuring
            classification error)
        :type nu: float
        :param eta_mul: initial setting of the learning rate (default 2.0)
        :type eta_mul: float
        """
        self._estimator = estimator
        self._constraints = constraints
        self._eps = eps
        self._T = T
        self._nu = nu
        self._eta_mul = eta_mul
        self._best_classifier = None
        self._classifiers = None

    def fit(self, X, y, **kwargs):
        """ Return a fair classifier under specified fairness constraints via
            exponentiated-gradient reduction.
        """
        # TODO: validate input data; unify between grid search and expgrad?
        if type(X) in [np.ndarray, list]:
            X_train = pd.DataFrame(X)
        else:
            X_train = X

        if type(y) in [np.ndarray, list]:
            y_train = pd.Series(y)
        else:
            y_train = y

        A = kwargs[_KW_SENSITIVE_FEATURES]

        n = X_train.shape[0]

        logger.debug("...Exponentiated Gradient STARTING")

        B = 1 / self._eps
        lagrangian = _Lagrangian(X_train, A, y_train, self._estimator, self._constraints,
                                 self._eps, B)

        theta = pd.Series(0, lagrangian.constraints.index)
        Qsum = pd.Series()
        lambdas = pd.DataFrame()
        gaps_EG = []
        gaps = []
        Qs = []

        last_regret_checked = _REGRET_CHECK_START_T
        last_gap = np.PINF
        for t in range(0, self._T):
            logger.debug("...iter=%03d" % t)

            # set lambdas for every constraint
            lambda_vec = B * np.exp(theta) / (1 + np.exp(theta).sum())
            lambdas[t] = lambda_vec
            lambda_EG = lambdas.mean(axis=1)

            # select classifier according to best_h method
            h, h_idx = lagrangian.best_h(lambda_vec)
            pred_h = h(X_train)

            if t == 0:
                if self._nu is None:
                    self._nu = _ACCURACY_MUL * (pred_h - y_train).abs().std() / np.sqrt(n)
                eta_min = self._nu / (2 * B)
                eta = self._eta_mul / B
                logger.debug("...eps=%.3f, B=%.1f, nu=%.6f, T=%d, eta_min=%.6f"
                             % (self._eps, B, self._nu, self._T, eta_min))

            if h_idx not in Qsum.index:
                Qsum.at[h_idx] = 0.0
            Qsum[h_idx] += 1.0
            gamma = lagrangian.gammas[h_idx]
            Q_EG = Qsum / Qsum.sum()
            result_EG = lagrangian.eval_gap(Q_EG, lambda_EG, self._nu)
            gap_EG = result_EG.gap()
            gaps_EG.append(gap_EG)

            if t == 0 or not _RUN_LP_STEP:
                gap_LP = np.PINF
            else:
                # saddle point optimization over the convex hull of
                # classifiers returned so far
                Q_LP, _, result_LP = lagrangian.solve_linprog(self._nu)
                gap_LP = result_LP.gap()

            # keep values from exponentiated gradient or linear programming
            if gap_EG < gap_LP:
                Qs.append(Q_EG)
                gaps.append(gap_EG)
            else:
                Qs.append(Q_LP)
                gaps.append(gap_LP)

            logger.debug("%seta=%.6f, L_low=%.3f, L=%.3f, L_high=%.3f"
                         ", gap=%.6f, disp=%.3f, err=%.3f, gap_LP=%.6f"
                         % (_INDENTATION, eta, result_EG.L_low,
                            result_EG.L, result_EG.L_high,
                            gap_EG, result_EG.gamma.max(),
                            result_EG.error, gap_LP))

            if (gaps[t] < self._nu) and (t >= _MIN_T):
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
            theta += eta * (gamma - self._eps)

        self._expgrad_result = self._format_results(gaps, Qs, lagrangian, B, eta_min)

        self._best_classifier = self._expgrad_result._best_classifier
        self._classifiers = self._expgrad_result._classifiers
        # TODO: figure out whether we should keep the remaining data of the result object

    def predict(self, X):
        positive_probs = self._best_classifier(X)
        return (positive_probs >= np.random.rand(len(positive_probs))) * 1

    def _pmf_predict(self, X):
        positive_probs = self._best_classifier(X)
        return np.concatenate((1-positive_probs, positive_probs), axis=1)

    def _format_results(self, gaps, Qs, lagrangian, B, eta_min):
        gaps_series = pd.Series(gaps)
        gaps_best = gaps_series[gaps_series <= gaps_series.min() + _PRECISION]
        best_t = gaps_best.index[-1]
        weights = Qs[best_t]
        hs = lagrangian.hs
        for h_idx in hs.index:
            if h_idx not in weights.index:
                weights.at[h_idx] = 0.0

        def best_classifier(X): return _mean_pred(X, hs, weights)
        best_gap = gaps[best_t]

        last_t = len(Qs) - 1

        result = ExponentiatedGradientResult(
            best_classifier,
            best_gap,
            lagrangian.classifiers,
            weights,
            last_t,
            best_t,
            lagrangian.n_oracle_calls)

        logger.debug("...eps=%.3f, B=%.1f, nu=%.6f, T=%d, eta_min=%.6f"
                     % (self._eps, B, self._nu, self._T, eta_min))
        logger.debug("...last_t=%d, best_t=%d, best_gap=%.6f, n_oracle_calls=%d, n_hs=%d"
                     % (last_t, best_t, best_gap, lagrangian.n_oracle_calls,
                        len(lagrangian.classifiers)))

        return result
