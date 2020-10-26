# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest
import sklearn.metrics as skm

import fairlearn.metrics as metrics

from .data_for_test import y_t, y_p, g_1, g_2, g_3, g_4


metric = [skm.recall_score,
          skm.precision_score,
          skm.accuracy_score,
          skm.balanced_accuracy_score]


class Test1m1sf0cf:
    # Single metric supplied as callable
    def _prepare(self, metric_fn):
        self.target = metrics.MetricFrame(metric_fn,
                                          y_t, y_p,
                                          sensitive_features=g_4)

        assert self.target.control_levels is None
        assert isinstance(self.target.sensitive_levels, list)
        assert (self.target.sensitive_levels == ['sensitive_feature_0'])

        self.overall = metric_fn(y_t, y_p)
        mask_p = (g_4 == 'pp')
        mask_q = (g_4 == 'q')
        self.metric_p = metric_fn(y_t[mask_p], y_p[mask_p])
        self.metric_q = metric_fn(y_t[mask_q], y_p[mask_q])

        self.mfn = metric_fn.__name__

    @pytest.mark.parametrize("metric_fn", metric)
    def test_min(self, metric_fn):
        self._prepare(metric_fn)

        target_mins = self.target.group_min()
        assert isinstance(target_mins, pd.Series)
        assert len(target_mins) == 1
        assert target_mins[self.mfn] == min(self.metric_p, self.metric_q)

    @pytest.mark.parametrize("metric_fn", metric)
    def test_max(self, metric_fn):
        self._prepare(metric_fn)

        target_maxes = self.target.group_max()
        assert isinstance(target_maxes, pd.Series)
        assert len(target_maxes) == 1
        assert target_maxes[self.mfn] == max(self.metric_p, self.metric_q)

    @pytest.mark.parametrize("metric_fn", metric)
    def test_difference_between_groups(self, metric_fn):
        self._prepare(metric_fn)

        target_diff = self.target.difference(method='between_groups')
        assert isinstance(target_diff, pd.Series)
        assert len(target_diff) == 1
        assert target_diff[self.mfn] == abs(self.metric_p - self.metric_q)

    @pytest.mark.parametrize("metric_fn", metric)
    def test_difference_to_overall(self, metric_fn):
        self._prepare(metric_fn)

        target_diff_overall = self.target.difference(method='to_overall')
        assert isinstance(target_diff_overall, pd.Series)
        assert len(target_diff_overall) == 1
        diffs_overall = [abs(self.metric_p-self.overall),
                         abs(self.metric_q-self.overall)]
        assert target_diff_overall[self.mfn] == max(diffs_overall)

    @pytest.mark.parametrize("metric_fn", metric)
    def test_ratio_between_groups(self, metric_fn):
        self._prepare(metric_fn)

        target_ratio = self.target.ratio(method='between_groups')
        assert isinstance(target_ratio, pd.Series)
        assert len(target_ratio) == 1
        assert target_ratio[self.mfn] == min(self.metric_p, self.metric_q) / \
            max(self.metric_p, self.metric_q)

    @pytest.mark.parametrize("metric_fn", metric)
    def test_ratio_to_overall(self, metric_fn):
        self._prepare(metric_fn)

        target_ratio_overall = self.target.ratio(method='to_overall')
        assert isinstance(target_ratio_overall, pd.Series)
        assert len(target_ratio_overall) == 1
        expected_ratio_overall = min(self.metric_p/self.overall,
                                     self.overall/self.metric_p,
                                     self.metric_q/self.overall,
                                     self.overall/self.metric_q)
        # Need to use approx, due to internal method of calculating
        # the correct ratio. Internally, MetricFrame computes a ratio
        # and takes the reciprocal if it's greater than 1
        assert target_ratio_overall[self.mfn] == pytest.approx(expected_ratio_overall,
                                                               rel=1e-10, abs=1e-16)


class Test1m1sf0cfFnDict:
    # Key difference is that the function is supplied as a dict
    def _prepare(self, metric_fn):
        self.mfn = "Random name"
        self.target = metrics.MetricFrame({self.mfn: metric_fn},
                                          y_t, y_p,
                                          sensitive_features=g_4)

        assert self.target.control_levels is None
        assert isinstance(self.target.sensitive_levels, list)
        assert (self.target.sensitive_levels == ['sensitive_feature_0'])

        self.overall = metric_fn(y_t, y_p)
        mask_p = (g_4 == 'pp')
        mask_q = (g_4 == 'q')
        self.metric_p = metric_fn(y_t[mask_p], y_p[mask_p])
        self.metric_q = metric_fn(y_t[mask_q], y_p[mask_q])

    @pytest.mark.parametrize("metric_fn", metric)
    def test_min(self, metric_fn):
        self._prepare(metric_fn)

        target_mins = self.target.group_min()
        assert isinstance(target_mins, pd.Series)
        assert len(target_mins) == 1
        assert target_mins[self.mfn] == min(self.metric_p, self.metric_q)

    @pytest.mark.parametrize("metric_fn", metric)
    def test_max(self, metric_fn):
        self._prepare(metric_fn)

        target_maxes = self.target.group_max()
        assert isinstance(target_maxes, pd.Series)
        assert len(target_maxes) == 1
        assert target_maxes[self.mfn] == max(self.metric_p, self.metric_q)

    @pytest.mark.parametrize("metric_fn", metric)
    def test_difference_between_groups(self, metric_fn):
        self._prepare(metric_fn)

        target_diff = self.target.difference(method='between_groups')
        assert isinstance(target_diff, pd.Series)
        assert len(target_diff) == 1
        assert target_diff[self.mfn] == abs(self.metric_p - self.metric_q)

    @pytest.mark.parametrize("metric_fn", metric)
    def test_difference_to_overall(self, metric_fn):
        self._prepare(metric_fn)

        target_diff_overall = self.target.difference(method='to_overall')
        assert isinstance(target_diff_overall, pd.Series)
        assert len(target_diff_overall) == 1
        diffs_overall = [abs(self.metric_p-self.overall),
                         abs(self.metric_q-self.overall)]
        assert target_diff_overall[self.mfn] == max(diffs_overall)

    @pytest.mark.parametrize("metric_fn", metric)
    def test_ratio_between_groups(self, metric_fn):
        self._prepare(metric_fn)

        target_ratio = self.target.ratio(method='between_groups')
        assert isinstance(target_ratio, pd.Series)
        assert len(target_ratio) == 1
        assert target_ratio[self.mfn] == min(self.metric_p, self.metric_q) / \
            max(self.metric_p, self.metric_q)

    @pytest.mark.parametrize("metric_fn", metric)
    def test_ratio_to_overall(self, metric_fn):
        self._prepare(metric_fn)

        target_ratio_overall = self.target.ratio(method='to_overall')
        assert isinstance(target_ratio_overall, pd.Series)
        assert len(target_ratio_overall) == 1
        expected_ratio_overall = min(self.metric_p/self.overall,
                                     self.overall/self.metric_p,
                                     self.metric_q/self.overall,
                                     self.overall/self.metric_q)
        # Need to use approx, due to internal method of calculating
        # the correct ratio. Internally, MetricFrame computes a ratio
        # and takes the reciprocal if it's greater than 1
        assert target_ratio_overall[self.mfn] == pytest.approx(expected_ratio_overall,
                                                               rel=1e-10, abs=1e-16)


class Test2m1sf0cf:
    def _prepare(self):
        fns = {'recall': skm.recall_score, 'prec': skm.precision_score}
        self.target = metrics.MetricFrame(fns,
                                          y_t, y_p,
                                          sensitive_features=pd.Series(data=g_4))

        assert self.target.control_levels is None
        assert isinstance(self.target.sensitive_levels, list)
        assert (self.target.sensitive_levels == ['sensitive_feature_0'])

        self.recall = skm.recall_score(y_t, y_p)
        self.prec = skm.precision_score(y_t, y_p)
        mask_p = (g_4 == 'pp')
        mask_q = (g_4 == 'q')
        self.recall_p = skm.recall_score(y_t[mask_p], y_p[mask_p])
        self.recall_q = skm.recall_score(y_t[mask_q], y_p[mask_q])
        self.prec_p = skm.precision_score(y_t[mask_p], y_p[mask_p])
        self.prec_q = skm.precision_score(y_t[mask_q], y_p[mask_q])

    def test_min(self):
        self._prepare()

        target_mins = self.target.group_min()
        assert isinstance(target_mins, pd.Series)
        assert len(target_mins) == 2
        assert target_mins['recall'] == min(self.recall_p, self.recall_q)
        assert target_mins['prec'] == min(self.prec_p, self.prec_q)

    def test_max(self):
        self._prepare()

        target_maxes = self.target.group_max()
        assert isinstance(target_maxes, pd.Series)
        assert target_maxes['recall'] == max(self.recall_p, self.recall_q)
        assert target_maxes['prec'] == max(self.prec_p, self.prec_q)

    def test_difference_between_groups(self):
        self._prepare()

        target_diffs = self.target.difference(method='between_groups')
        assert isinstance(target_diffs, pd.Series)
        assert len(target_diffs) == 2
        assert target_diffs['recall'] == abs(self.recall_p - self.recall_q)
        assert target_diffs['prec'] == abs(self.prec_p - self.prec_q)

    def test_difference_to_overall(self):
        self._prepare()

        target_diff_overall = self.target.difference(method='to_overall')
        assert isinstance(target_diff_overall, pd.Series)
        assert len(target_diff_overall) == 2
        recall_diffs_overall = [abs(self.recall_p-self.recall), abs(self.recall_q-self.recall)]
        prec_diffs_overall = [abs(self.prec_p-self.prec), abs(self.prec_q-self.prec)]
        assert target_diff_overall['recall'] == max(recall_diffs_overall)
        assert target_diff_overall['prec'] == max(prec_diffs_overall)

    def test_ratio_between_groups(self):
        self._prepare()

        target_ratio = self.target.ratio(method='between_groups')
        assert isinstance(target_ratio, pd.Series)
        assert len(target_ratio) == 2
        assert target_ratio['recall'] == min(self.recall_p, self.recall_q) / \
            max(self.recall_p, self.recall_q)
        assert target_ratio['prec'] == min(self.prec_p/self.prec_q, self.prec_q/self.prec_p)

    def test_ratio_to_overall(self):
        self._prepare()

        target_ratio_overall = self.target.ratio(method='to_overall')
        assert isinstance(target_ratio_overall, pd.Series)
        assert len(target_ratio_overall) == 2
        recall_ratio_overall = min(self.recall_p/self.recall, self.recall/self.recall_p,
                                   self.recall_q/self.recall, self.recall/self.recall_q)
        prec_ratio_overall = min(self.prec_p/self.prec, self.prec/self.prec_p,
                                 self.prec_q/self.prec, self.prec/self.prec_q)
        # Need pytest.approx because ratio() doesn't do quite the same
        # calculations as above. Specifically, rather than doing both
        # preq_q/prec and prec/prec_q, it only calculates the former,
        # and inverts it if the result is greater than unity
        assert target_ratio_overall['recall'] == pytest.approx(recall_ratio_overall,
                                                               rel=1e-10, abs=1e-16)
        assert target_ratio_overall['prec'] == pytest.approx(prec_ratio_overall,
                                                             rel=1e-10, abs=1e-16)


class Test1m1cf1sf:
    # Metric function supplied in callable
    def _prepare(self, metric_fn):
        self.target = metrics.MetricFrame(metric_fn,
                                          y_t, y_p,
                                          sensitive_features=pd.Series(data=g_2, name='sf0'),
                                          control_features=pd.Series(data=g_3, name='cf0'))

        assert isinstance(self.target.control_levels, list)
        assert (self.target.control_levels == ['cf0'])
        assert isinstance(self.target.sensitive_levels, list)
        assert (self.target.sensitive_levels == ['sf0'])

        mask_f = (g_2 == 'f')
        mask_g = (g_2 == 'g')
        mask_k = (g_3 == 'kk')
        mask_m = (g_3 == 'm')

        mask_k_f = np.logical_and(mask_k, mask_f)
        mask_k_g = np.logical_and(mask_k, mask_g)
        mask_m_f = np.logical_and(mask_m, mask_f)
        mask_m_g = np.logical_and(mask_m, mask_g)
        self.metric_k = metric_fn(y_t[mask_k], y_p[mask_k])
        self.metric_m = metric_fn(y_t[mask_m], y_p[mask_m])
        self.metric_k_f = metric_fn(y_t[mask_k_f], y_p[mask_k_f])
        self.metric_m_f = metric_fn(y_t[mask_m_f], y_p[mask_m_f])
        self.metric_k_g = metric_fn(y_t[mask_k_g], y_p[mask_k_g])
        self.metric_m_g = metric_fn(y_t[mask_m_g], y_p[mask_m_g])
        self.metric_k_arr = [self.metric_k_f, self.metric_k_g]
        self.metric_m_arr = [self.metric_m_f, self.metric_m_g]

        self.mfn = metric_fn.__name__

    @pytest.mark.parametrize("metric_fn", metric)
    def test_min(self, metric_fn):
        self._prepare(metric_fn)

        target_mins = self.target.group_min()
        assert isinstance(target_mins, pd.DataFrame)
        assert target_mins.shape == (2, 1)
        assert target_mins[self.mfn]['kk'] == min(self.metric_k_arr)
        assert target_mins[self.mfn]['m'] == min(self.metric_m_arr)

    @pytest.mark.parametrize("metric_fn", metric)
    def test_max(self, metric_fn):
        self._prepare(metric_fn)

        target_maxs = self.target.group_max()
        assert isinstance(target_maxs, pd.DataFrame)
        assert target_maxs.shape == (2, 1)
        assert target_maxs[self.mfn]['kk'] == max(self.metric_k_arr)
        assert target_maxs[self.mfn]['m'] == max(self.metric_m_arr)

    @pytest.mark.parametrize("metric_fn", metric)
    def test_difference_between_groups(self, metric_fn):
        self._prepare(metric_fn)

        target_diff = self.target.difference(method='between_groups')
        assert isinstance(target_diff, pd.DataFrame)
        assert target_diff.shape == (2, 1)
        assert target_diff[self.mfn]['kk'] == max(self.metric_k_arr)-min(self.metric_k_arr)
        assert target_diff[self.mfn]['m'] == max(self.metric_m_arr)-min(self.metric_m_arr)

    @pytest.mark.parametrize("metric_fn", metric)
    def test_difference_to_overall(self, metric_fn):
        self._prepare(metric_fn)

        target_diff_overall = self.target.difference(method='to_overall')
        assert isinstance(target_diff_overall, pd.DataFrame)
        assert target_diff_overall.shape == (2, 1)
        k_diffs_overall = [abs(self.metric_k_f-self.metric_k),
                           abs(self.metric_k_g-self.metric_k)]
        m_diffs_overall = [abs(self.metric_m_f-self.metric_m),
                           abs(self.metric_m_g-self.metric_m)]
        assert target_diff_overall[self.mfn]['kk'] == max(k_diffs_overall)
        assert target_diff_overall[self.mfn]['m'] == max(m_diffs_overall)

    @pytest.mark.parametrize("metric_fn", metric)
    def test_ratio_between_groups(self, metric_fn):
        self._prepare(metric_fn)

        target_ratio = self.target.ratio(method='between_groups')
        assert isinstance(target_ratio, pd.DataFrame)
        assert target_ratio.shape == (2, 1)
        assert target_ratio[self.mfn]['kk'] == min(self.metric_k_arr)/max(self.metric_k_arr)
        assert target_ratio[self.mfn]['m'] == min(self.metric_m_arr)/max(self.metric_m_arr)

    @pytest.mark.parametrize("metric_fn", metric)
    def test_ratio_to_overall(self, metric_fn):
        self._prepare(metric_fn)

        target_ratio_overall = self.target.ratio(method='to_overall')
        assert isinstance(target_ratio_overall, pd.DataFrame)
        assert target_ratio_overall.shape == (2, 1)
        k_ratios_overall = [x/self.metric_k for x in self.metric_k_arr] + \
            [self.metric_k/x for x in self.metric_k_arr]
        m_ratios_overall = [x/self.metric_m for x in self.metric_m_arr] + \
            [self.metric_m/x for x in self.metric_m_arr]
        # Ratio to overall is forced to be <1 in a slightly different way
        # internally, so have to use pytest.approx
        assert target_ratio_overall[self.mfn]['kk'] == pytest.approx(min(k_ratios_overall),
                                                                     rel=1e-10, abs=1e-16)
        assert target_ratio_overall[self.mfn]['m'] == pytest.approx(min(m_ratios_overall),
                                                                    rel=1e-10, abs=1e-16)


class Test1m1cf1sfFnDict:
    # Key difference is that the metric function is supplied in a dict
    def _prepare(self, metric_fn):
        self.mfn = "Some name"
        self.target = metrics.MetricFrame({self.mfn: metric_fn},
                                          y_t, y_p,
                                          sensitive_features=pd.Series(data=g_2, name='sf0'),
                                          control_features=pd.Series(data=g_3, name='cf0'))

        assert isinstance(self.target.control_levels, list)
        assert (self.target.control_levels == ['cf0'])
        assert isinstance(self.target.sensitive_levels, list)
        assert (self.target.sensitive_levels == ['sf0'])

        mask_f = (g_2 == 'f')
        mask_g = (g_2 == 'g')
        mask_k = (g_3 == 'kk')
        mask_m = (g_3 == 'm')

        mask_k_f = np.logical_and(mask_k, mask_f)
        mask_k_g = np.logical_and(mask_k, mask_g)
        mask_m_f = np.logical_and(mask_m, mask_f)
        mask_m_g = np.logical_and(mask_m, mask_g)
        self.metric_k = metric_fn(y_t[mask_k], y_p[mask_k])
        self.metric_m = metric_fn(y_t[mask_m], y_p[mask_m])
        self.metric_k_f = metric_fn(y_t[mask_k_f], y_p[mask_k_f])
        self.metric_m_f = metric_fn(y_t[mask_m_f], y_p[mask_m_f])
        self.metric_k_g = metric_fn(y_t[mask_k_g], y_p[mask_k_g])
        self.metric_m_g = metric_fn(y_t[mask_m_g], y_p[mask_m_g])
        self.metric_k_arr = [self.metric_k_f, self.metric_k_g]
        self.metric_m_arr = [self.metric_m_f, self.metric_m_g]

    @pytest.mark.parametrize("metric_fn", metric)
    def test_min(self, metric_fn):
        self._prepare(metric_fn)

        target_mins = self.target.group_min()
        assert isinstance(target_mins, pd.DataFrame)
        assert target_mins.shape == (2, 1)
        assert target_mins[self.mfn]['kk'] == min(self.metric_k_arr)
        assert target_mins[self.mfn]['m'] == min(self.metric_m_arr)

    @pytest.mark.parametrize("metric_fn", metric)
    def test_max(self, metric_fn):
        self._prepare(metric_fn)

        target_maxs = self.target.group_max()
        assert isinstance(target_maxs, pd.DataFrame)
        assert target_maxs.shape == (2, 1)
        assert target_maxs[self.mfn]['kk'] == max(self.metric_k_arr)
        assert target_maxs[self.mfn]['m'] == max(self.metric_m_arr)

    @pytest.mark.parametrize("metric_fn", metric)
    def test_difference_between_groups(self, metric_fn):
        self._prepare(metric_fn)

        target_diff = self.target.difference(method='between_groups')
        assert isinstance(target_diff, pd.DataFrame)
        assert target_diff.shape == (2, 1)
        assert target_diff[self.mfn]['kk'] == max(self.metric_k_arr)-min(self.metric_k_arr)
        assert target_diff[self.mfn]['m'] == max(self.metric_m_arr)-min(self.metric_m_arr)

    @pytest.mark.parametrize("metric_fn", metric)
    def test_difference_to_overall(self, metric_fn):
        self._prepare(metric_fn)

        target_diff_overall = self.target.difference(method='to_overall')
        assert isinstance(target_diff_overall, pd.DataFrame)
        assert target_diff_overall.shape == (2, 1)
        k_diffs_overall = [abs(self.metric_k_f-self.metric_k),
                           abs(self.metric_k_g-self.metric_k)]
        m_diffs_overall = [abs(self.metric_m_f-self.metric_m),
                           abs(self.metric_m_g-self.metric_m)]
        assert target_diff_overall[self.mfn]['kk'] == max(k_diffs_overall)
        assert target_diff_overall[self.mfn]['m'] == max(m_diffs_overall)

    @pytest.mark.parametrize("metric_fn", metric)
    def test_ratio_between_groups(self, metric_fn):
        self._prepare(metric_fn)

        target_ratio = self.target.ratio(method='between_groups')
        assert isinstance(target_ratio, pd.DataFrame)
        assert target_ratio.shape == (2, 1)
        assert target_ratio[self.mfn]['kk'] == min(self.metric_k_arr)/max(self.metric_k_arr)
        assert target_ratio[self.mfn]['m'] == min(self.metric_m_arr)/max(self.metric_m_arr)

    @pytest.mark.parametrize("metric_fn", metric)
    def test_ratio_to_overall(self, metric_fn):
        self._prepare(metric_fn)

        target_ratio_overall = self.target.ratio(method='to_overall')
        assert isinstance(target_ratio_overall, pd.DataFrame)
        assert target_ratio_overall.shape == (2, 1)
        k_ratios_overall = [x/self.metric_k for x in self.metric_k_arr] + \
            [self.metric_k/x for x in self.metric_k_arr]
        m_ratios_overall = [x/self.metric_m for x in self.metric_m_arr] + \
            [self.metric_m/x for x in self.metric_m_arr]
        # Ratio to overall is forced to be <1 in a slightly different way
        # internally, so have to use pytest.approx
        assert target_ratio_overall[self.mfn]['kk'] == pytest.approx(min(k_ratios_overall),
                                                                     rel=1e-10, abs=1e-16)
        assert target_ratio_overall[self.mfn]['m'] == pytest.approx(min(m_ratios_overall),
                                                                    rel=1e-10, abs=1e-16)


class Test1m1sf2cf:
    # Single metric, supplied as callable
    def _prepare(self, metric_fn):
        self.target = metrics.MetricFrame(metric_fn,
                                          y_t, y_p,
                                          sensitive_features=list(g_2),
                                          control_features=np.stack([g_3, g_1], axis=1))

        assert isinstance(self.target.control_levels, list)
        assert (self.target.control_levels == ['control_feature_0', 'control_feature_1'])
        assert isinstance(self.target.sensitive_levels, list)
        assert (self.target.sensitive_levels == ['sensitive_feature_0'])

        # Check we have correct return types
        assert isinstance(self.target.overall, pd.DataFrame)
        assert isinstance(self.target.by_group, pd.DataFrame)

        mask_a = (g_1 == 'aa')
        mask_b = (g_1 == 'ba')
        mask_f = (g_2 == 'f')
        mask_g = (g_2 == 'g')
        mask_k = (g_3 == 'kk')
        mask_m = (g_3 == 'm')

        mask_k_a = np.logical_and(mask_k, mask_a)
        mask_k_b = np.logical_and(mask_k, mask_b)
        mask_m_a = np.logical_and(mask_m, mask_a)
        mask_m_b = np.logical_and(mask_m, mask_b)
        mask_k_a_f = np.logical_and(mask_k_a, mask_f)
        mask_k_a_g = np.logical_and(mask_k_a, mask_g)
        mask_k_b_f = np.logical_and(mask_k_b, mask_f)
        mask_k_b_g = np.logical_and(mask_k_b, mask_g)
        mask_m_a_f = np.logical_and(mask_m_a, mask_f)
        mask_m_a_g = np.logical_and(mask_m_a, mask_g)
        mask_m_b_f = np.logical_and(mask_m_b, mask_f)
        mask_m_b_g = np.logical_and(mask_m_b, mask_g)

        self.metric_k_a = metric_fn(y_t[mask_k_a], y_p[mask_k_a])
        self.metric_k_b = metric_fn(y_t[mask_k_b], y_p[mask_k_b])
        self.metric_m_a = metric_fn(y_t[mask_m_a], y_p[mask_m_a])
        self.metric_m_b = metric_fn(y_t[mask_m_b], y_p[mask_m_b])
        self.metric_k_a_f = metric_fn(y_t[mask_k_a_f], y_p[mask_k_a_f])
        self.metric_k_a_g = metric_fn(y_t[mask_k_a_g], y_p[mask_k_a_g])
        self.metric_k_b_f = metric_fn(y_t[mask_k_b_f], y_p[mask_k_b_f])
        self.metric_k_b_g = metric_fn(y_t[mask_k_b_g], y_p[mask_k_b_g])
        self.metric_m_a_f = metric_fn(y_t[mask_m_a_f], y_p[mask_m_a_f])
        self.metric_m_a_g = metric_fn(y_t[mask_m_a_g], y_p[mask_m_a_g])
        self.metric_m_b_f = metric_fn(y_t[mask_m_b_f], y_p[mask_m_b_f])
        self.metric_m_b_g = metric_fn(y_t[mask_m_b_g], y_p[mask_m_b_g])

        self.metric_k_a_arr = [self.metric_k_a_f, self.metric_k_a_g]
        self.metric_k_b_arr = [self.metric_k_b_f, self.metric_k_b_g]
        self.metric_m_a_arr = [self.metric_m_a_f, self.metric_m_a_g]
        self.metric_m_b_arr = [self.metric_m_b_f, self.metric_m_b_g]

        self.mfn = metric_fn.__name__

    @pytest.mark.parametrize("metric_fn", metric)
    def test_min(self, metric_fn):
        self._prepare(metric_fn)

        target_mins = self.target.group_min()
        assert isinstance(target_mins, pd.DataFrame)
        assert target_mins.shape == (4, 1)
        assert target_mins[self.mfn][('kk', 'aa')] == min(self.metric_k_a_arr)
        assert target_mins[self.mfn][('kk', 'ba')] == min(self.metric_k_b_arr)
        assert target_mins[self.mfn][('m', 'aa')] == min(self.metric_m_a_arr)
        assert target_mins[self.mfn][('m', 'ba')] == min(self.metric_m_b_arr)

    @pytest.mark.parametrize("metric_fn", metric)
    def test_max(self, metric_fn):
        self._prepare(metric_fn)

        target_maxs = self.target.group_max()
        assert isinstance(target_maxs, pd.DataFrame)
        assert target_maxs.shape == (4, 1)
        assert target_maxs[self.mfn][('kk', 'aa')] == max(self.metric_k_a_arr)
        assert target_maxs[self.mfn][('kk', 'ba')] == max(self.metric_k_b_arr)
        assert target_maxs[self.mfn][('m', 'aa')] == max(self.metric_m_a_arr)
        assert target_maxs[self.mfn][('m', 'ba')] == max(self.metric_m_b_arr)

    @pytest.mark.parametrize("metric_fn", metric)
    def test_difference_between_groups(self, metric_fn):
        self._prepare(metric_fn)

        diffs = self.target.difference(method='between_groups')
        assert isinstance(diffs, pd.DataFrame)
        assert diffs.shape == (4, 1)
        assert diffs[self.mfn][('kk', 'aa')] == max(self.metric_k_a_arr) - min(self.metric_k_a_arr)
        assert diffs[self.mfn][('kk', 'ba')] == max(self.metric_k_b_arr) - min(self.metric_k_b_arr)
        assert diffs[self.mfn][('m', 'aa')] == max(self.metric_m_a_arr) - min(self.metric_m_a_arr)
        assert diffs[self.mfn][('m', 'ba')] == max(self.metric_m_b_arr) - min(self.metric_m_b_arr)

    @pytest.mark.parametrize("metric_fn", metric)
    def test_difference_to_overall(self, metric_fn):
        self._prepare(metric_fn)

        diffs_overall = self.target.difference(method='to_overall')
        assert isinstance(diffs_overall, pd.DataFrame)
        assert diffs_overall.shape == (4, 1)
        diff_overall_k_a = max([abs(x-self.metric_k_a) for x in self.metric_k_a_arr])
        diff_overall_k_b = max([abs(x-self.metric_k_b) for x in self.metric_k_b_arr])
        diff_overall_m_a = max([abs(x-self.metric_m_a) for x in self.metric_m_a_arr])
        diff_overall_m_b = max([abs(x-self.metric_m_b) for x in self.metric_m_b_arr])

        assert diffs_overall[self.mfn][('kk', 'aa')] == diff_overall_k_a
        assert diffs_overall[self.mfn][('kk', 'ba')] == diff_overall_k_b
        assert diffs_overall[self.mfn][('m', 'aa')] == diff_overall_m_a
        assert diffs_overall[self.mfn][('m', 'ba')] == diff_overall_m_b

    @pytest.mark.parametrize("metric_fn", metric)
    def test_ratio_between_groups(self, metric_fn):
        self._prepare(metric_fn)

        ratios = self.target.ratio(method='between_groups')
        assert isinstance(ratios, pd.DataFrame)
        assert ratios.shape == (4, 1)
        assert ratios[self.mfn][('kk', 'aa')] == \
            min(self.metric_k_a_arr) / max(self.metric_k_a_arr)
        assert ratios[self.mfn][('kk', 'ba')] == \
            min(self.metric_k_b_arr) / max(self.metric_k_b_arr)
        assert ratios[self.mfn][('m', 'aa')] == \
            min(self.metric_m_a_arr) / max(self.metric_m_a_arr)
        assert ratios[self.mfn][('m', 'ba')] == \
            min(self.metric_m_b_arr) / max(self.metric_m_b_arr)

    @pytest.mark.parametrize("metric_fn", metric)
    def test_ratio_to_overall(self, metric_fn):
        self._prepare(metric_fn)

        ratios_overall = self.target.ratio(method='to_overall')
        assert isinstance(ratios_overall, pd.DataFrame)
        assert ratios_overall.shape == (4, 1)
        ratio_overall_k_a = [x/self.metric_k_a for x in self.metric_k_a_arr] + \
            [self.metric_k_a/x for x in self.metric_k_a_arr]
        ratio_overall_k_b = [x/self.metric_k_b for x in self.metric_k_b_arr] + \
            [self.metric_k_b/x for x in self.metric_k_b_arr]
        ratio_overall_m_a = [x/self.metric_m_a for x in self.metric_m_a_arr] + \
            [self.metric_m_a/x for x in self.metric_m_a_arr]
        ratio_overall_m_b = [x/self.metric_m_b for x in self.metric_m_b_arr] + \
            [self.metric_m_b/x for x in self.metric_m_b_arr]
        assert ratios_overall[self.mfn][('kk', 'aa')] == min(ratio_overall_k_a)
        assert ratios_overall[self.mfn][('kk', 'ba')] == min(ratio_overall_k_b)
        assert ratios_overall[self.mfn][('m', 'aa')] == min(ratio_overall_m_a)
        assert ratios_overall[self.mfn][('m', 'ba')] == pytest.approx(min(ratio_overall_m_b),
                                                                      rel=1e-10, abs=1e-16)


class Test1m1sf2cfFnDict:
    # Single metric, supplied as dict
    def _prepare(self, metric_fn):
        self.mfn = "Random name"
        self.target = metrics.MetricFrame({self.mfn: metric_fn},
                                          y_t, y_p,
                                          sensitive_features=list(g_2),
                                          control_features=np.stack([g_3, g_1], axis=1))

        assert isinstance(self.target.control_levels, list)
        assert (self.target.control_levels == ['control_feature_0', 'control_feature_1'])
        assert isinstance(self.target.sensitive_levels, list)
        assert (self.target.sensitive_levels == ['sensitive_feature_0'])

        # Check we have correct return types
        assert isinstance(self.target.overall, pd.DataFrame)
        assert isinstance(self.target.by_group, pd.DataFrame)

        mask_a = (g_1 == 'aa')
        mask_b = (g_1 == 'ba')
        mask_f = (g_2 == 'f')
        mask_g = (g_2 == 'g')
        mask_k = (g_3 == 'kk')
        mask_m = (g_3 == 'm')

        mask_k_a = np.logical_and(mask_k, mask_a)
        mask_k_b = np.logical_and(mask_k, mask_b)
        mask_m_a = np.logical_and(mask_m, mask_a)
        mask_m_b = np.logical_and(mask_m, mask_b)
        mask_k_a_f = np.logical_and(mask_k_a, mask_f)
        mask_k_a_g = np.logical_and(mask_k_a, mask_g)
        mask_k_b_f = np.logical_and(mask_k_b, mask_f)
        mask_k_b_g = np.logical_and(mask_k_b, mask_g)
        mask_m_a_f = np.logical_and(mask_m_a, mask_f)
        mask_m_a_g = np.logical_and(mask_m_a, mask_g)
        mask_m_b_f = np.logical_and(mask_m_b, mask_f)
        mask_m_b_g = np.logical_and(mask_m_b, mask_g)

        self.metric_k_a = metric_fn(y_t[mask_k_a], y_p[mask_k_a])
        self.metric_k_b = metric_fn(y_t[mask_k_b], y_p[mask_k_b])
        self.metric_m_a = metric_fn(y_t[mask_m_a], y_p[mask_m_a])
        self.metric_m_b = metric_fn(y_t[mask_m_b], y_p[mask_m_b])
        self.metric_k_a_f = metric_fn(y_t[mask_k_a_f], y_p[mask_k_a_f])
        self.metric_k_a_g = metric_fn(y_t[mask_k_a_g], y_p[mask_k_a_g])
        self.metric_k_b_f = metric_fn(y_t[mask_k_b_f], y_p[mask_k_b_f])
        self.metric_k_b_g = metric_fn(y_t[mask_k_b_g], y_p[mask_k_b_g])
        self.metric_m_a_f = metric_fn(y_t[mask_m_a_f], y_p[mask_m_a_f])
        self.metric_m_a_g = metric_fn(y_t[mask_m_a_g], y_p[mask_m_a_g])
        self.metric_m_b_f = metric_fn(y_t[mask_m_b_f], y_p[mask_m_b_f])
        self.metric_m_b_g = metric_fn(y_t[mask_m_b_g], y_p[mask_m_b_g])

        self.metric_k_a_arr = [self.metric_k_a_f, self.metric_k_a_g]
        self.metric_k_b_arr = [self.metric_k_b_f, self.metric_k_b_g]
        self.metric_m_a_arr = [self.metric_m_a_f, self.metric_m_a_g]
        self.metric_m_b_arr = [self.metric_m_b_f, self.metric_m_b_g]

    @pytest.mark.parametrize("metric_fn", metric)
    def test_min(self, metric_fn):
        self._prepare(metric_fn)

        target_mins = self.target.group_min()
        assert isinstance(target_mins, pd.DataFrame)
        assert target_mins.shape == (4, 1)
        assert target_mins[self.mfn][('kk', 'aa')] == min(self.metric_k_a_arr)
        assert target_mins[self.mfn][('kk', 'ba')] == min(self.metric_k_b_arr)
        assert target_mins[self.mfn][('m', 'aa')] == min(self.metric_m_a_arr)
        assert target_mins[self.mfn][('m', 'ba')] == min(self.metric_m_b_arr)

    @pytest.mark.parametrize("metric_fn", metric)
    def test_max(self, metric_fn):
        self._prepare(metric_fn)

        target_maxs = self.target.group_max()
        assert isinstance(target_maxs, pd.DataFrame)
        assert target_maxs.shape == (4, 1)
        assert target_maxs[self.mfn][('kk', 'aa')] == max(self.metric_k_a_arr)
        assert target_maxs[self.mfn][('kk', 'ba')] == max(self.metric_k_b_arr)
        assert target_maxs[self.mfn][('m', 'aa')] == max(self.metric_m_a_arr)
        assert target_maxs[self.mfn][('m', 'ba')] == max(self.metric_m_b_arr)

    @pytest.mark.parametrize("metric_fn", metric)
    def test_difference_between_groups(self, metric_fn):
        self._prepare(metric_fn)

        diffs = self.target.difference(method='between_groups')
        assert isinstance(diffs, pd.DataFrame)
        assert diffs.shape == (4, 1)
        assert diffs[self.mfn][('kk', 'aa')] == max(self.metric_k_a_arr) - min(self.metric_k_a_arr)
        assert diffs[self.mfn][('kk', 'ba')] == max(self.metric_k_b_arr) - min(self.metric_k_b_arr)
        assert diffs[self.mfn][('m', 'aa')] == max(self.metric_m_a_arr) - min(self.metric_m_a_arr)
        assert diffs[self.mfn][('m', 'ba')] == max(self.metric_m_b_arr) - min(self.metric_m_b_arr)

    @pytest.mark.parametrize("metric_fn", metric)
    def test_difference_to_overall(self, metric_fn):
        self._prepare(metric_fn)

        diffs_overall = self.target.difference(method='to_overall')
        assert isinstance(diffs_overall, pd.DataFrame)
        assert diffs_overall.shape == (4, 1)
        diff_overall_k_a = max([abs(x-self.metric_k_a) for x in self.metric_k_a_arr])
        diff_overall_k_b = max([abs(x-self.metric_k_b) for x in self.metric_k_b_arr])
        diff_overall_m_a = max([abs(x-self.metric_m_a) for x in self.metric_m_a_arr])
        diff_overall_m_b = max([abs(x-self.metric_m_b) for x in self.metric_m_b_arr])

        assert diffs_overall[self.mfn][('kk', 'aa')] == diff_overall_k_a
        assert diffs_overall[self.mfn][('kk', 'ba')] == diff_overall_k_b
        assert diffs_overall[self.mfn][('m', 'aa')] == diff_overall_m_a
        assert diffs_overall[self.mfn][('m', 'ba')] == diff_overall_m_b

    @pytest.mark.parametrize("metric_fn", metric)
    def test_ratio_between_groups(self, metric_fn):
        self._prepare(metric_fn)

        ratios = self.target.ratio(method='between_groups')
        assert isinstance(ratios, pd.DataFrame)
        assert ratios.shape == (4, 1)
        assert ratios[self.mfn][('kk', 'aa')] == \
            min(self.metric_k_a_arr) / max(self.metric_k_a_arr)
        assert ratios[self.mfn][('kk', 'ba')] == \
            min(self.metric_k_b_arr) / max(self.metric_k_b_arr)
        assert ratios[self.mfn][('m', 'aa')] == \
            min(self.metric_m_a_arr) / max(self.metric_m_a_arr)
        assert ratios[self.mfn][('m', 'ba')] == \
            min(self.metric_m_b_arr) / max(self.metric_m_b_arr)

    @pytest.mark.parametrize("metric_fn", metric)
    def test_ratio_to_overall(self, metric_fn):
        self._prepare(metric_fn)

        ratios_overall = self.target.ratio(method='to_overall')
        assert isinstance(ratios_overall, pd.DataFrame)
        assert ratios_overall.shape == (4, 1)
        ratio_overall_k_a = [x/self.metric_k_a for x in self.metric_k_a_arr] + \
            [self.metric_k_a/x for x in self.metric_k_a_arr]
        ratio_overall_k_b = [x/self.metric_k_b for x in self.metric_k_b_arr] + \
            [self.metric_k_b/x for x in self.metric_k_b_arr]
        ratio_overall_m_a = [x/self.metric_m_a for x in self.metric_m_a_arr] + \
            [self.metric_m_a/x for x in self.metric_m_a_arr]
        ratio_overall_m_b = [x/self.metric_m_b for x in self.metric_m_b_arr] + \
            [self.metric_m_b/x for x in self.metric_m_b_arr]
        assert ratios_overall[self.mfn][('kk', 'aa')] == min(ratio_overall_k_a)
        assert ratios_overall[self.mfn][('kk', 'ba')] == min(ratio_overall_k_b)
        assert ratios_overall[self.mfn][('m', 'aa')] == min(ratio_overall_m_a)
        assert ratios_overall[self.mfn][('m', 'ba')] == pytest.approx(min(ratio_overall_m_b),
                                                                      rel=1e-10, abs=1e-16)


class Test2m1sf1cf:
    def _prepare(self):
        fns = {'recall': skm.recall_score, 'prec': skm.precision_score}
        self.target = metrics.MetricFrame(fns,
                                          y_t, y_p,
                                          sensitive_features=g_2,
                                          control_features=g_3)

        assert isinstance(self.target.control_levels, list)
        assert (self.target.control_levels == ['control_feature_0'])
        assert isinstance(self.target.sensitive_levels, list)
        assert (self.target.sensitive_levels == ['sensitive_feature_0'])

        # Check we have correct return types
        assert isinstance(self.target.overall, pd.DataFrame)
        assert isinstance(self.target.by_group, pd.DataFrame)

        mask_f = (g_2 == 'f')
        mask_g = (g_2 == 'g')
        mask_k = (g_3 == 'kk')
        mask_m = (g_3 == 'm')

        mask_k_f = np.logical_and(mask_k, mask_f)
        mask_k_g = np.logical_and(mask_k, mask_g)
        mask_m_f = np.logical_and(mask_m, mask_f)
        mask_m_g = np.logical_and(mask_m, mask_g)
        self.recall_k = skm.recall_score(y_t[mask_k], y_p[mask_k])
        self.recall_m = skm.recall_score(y_t[mask_m], y_p[mask_m])
        self.recall_k_f = skm.recall_score(y_t[mask_k_f], y_p[mask_k_f])
        self.recall_m_f = skm.recall_score(y_t[mask_m_f], y_p[mask_m_f])
        self.recall_k_g = skm.recall_score(y_t[mask_k_g], y_p[mask_k_g])
        self.recall_m_g = skm.recall_score(y_t[mask_m_g], y_p[mask_m_g])
        self.recall_k_arr = [self.recall_k_f, self.recall_k_g]
        self.recall_m_arr = [self.recall_m_f, self.recall_m_g]
        self.precision_k = skm.precision_score(y_t[mask_k], y_p[mask_k])
        self.precision_m = skm.precision_score(y_t[mask_m], y_p[mask_m])
        self.precision_k_f = skm.precision_score(y_t[mask_k_f], y_p[mask_k_f])
        self.precision_m_f = skm.precision_score(y_t[mask_m_f], y_p[mask_m_f])
        self.precision_k_g = skm.precision_score(y_t[mask_k_g], y_p[mask_k_g])
        self.precision_m_g = skm.precision_score(y_t[mask_m_g], y_p[mask_m_g])
        self.precision_k_arr = [self.precision_k_f, self.precision_k_g]
        self.precision_m_arr = [self.precision_m_f, self.precision_m_g]

    def test_min(self):
        self._prepare()

        target_mins = self.target.group_min()
        assert isinstance(target_mins, pd.DataFrame)
        assert target_mins.shape == (2, 2)
        assert target_mins['recall']['kk'] == min(self.recall_k_arr)
        assert target_mins['recall']['m'] == min(self.recall_m_arr)
        assert target_mins['prec']['kk'] == min(self.precision_k_arr)
        assert target_mins['prec']['m'] == min(self.precision_m_arr)

    def test_max(self):
        self._prepare()

        target_maxs = self.target.group_max()
        assert isinstance(target_maxs, pd.DataFrame)
        assert target_maxs.shape == (2, 2)
        assert target_maxs['recall']['kk'] == max(self.recall_k_arr)
        assert target_maxs['recall']['m'] == max(self.recall_m_arr)
        assert target_maxs['prec']['kk'] == max(self.precision_k_arr)
        assert target_maxs['prec']['m'] == max(self.precision_m_arr)

    def test_difference_between_groups(self):
        self._prepare()

        diffs = self.target.difference(method='between_groups')
        assert isinstance(diffs, pd.DataFrame)
        assert diffs.shape == (2, 2)
        assert diffs['recall']['kk'] == max(self.recall_k_arr) - min(self.recall_k_arr)
        assert diffs['recall']['m'] == max(self.recall_m_arr) - min(self.recall_m_arr)
        assert diffs['prec']['kk'] == max(self.precision_k_arr) - min(self.precision_k_arr)
        assert diffs['prec']['m'] == max(self.precision_m_arr) - min(self.precision_m_arr)

    def test_difference_to_overall(self):
        self._prepare()

        diffs_overall = self.target.difference(method='to_overall')
        assert isinstance(diffs_overall, pd.DataFrame)
        assert diffs_overall.shape == (2, 2)
        recall_k_overall = max([abs(x-self.recall_k) for x in self.recall_k_arr])
        recall_m_overall = max([abs(x-self.recall_m) for x in self.recall_m_arr])
        precision_k_overall = max([abs(x-self.precision_k) for x in self.precision_k_arr])
        precision_m_overall = max([abs(x-self.precision_m) for x in self.precision_m_arr])
        assert diffs_overall['recall']['kk'] == recall_k_overall
        assert diffs_overall['recall']['m'] == recall_m_overall
        assert diffs_overall['prec']['kk'] == precision_k_overall
        assert diffs_overall['prec']['m'] == precision_m_overall

    def test_ratio_between_groups(self):
        self._prepare()

        ratios = self.target.ratio(method='between_groups')
        assert isinstance(ratios, pd.DataFrame)
        assert ratios.shape == (2, 2)
        assert ratios['recall']['kk'] == min(self.recall_k_arr) / max(self.recall_k_arr)
        assert ratios['recall']['m'] == min(self.recall_m_arr) / max(self.recall_m_arr)
        assert ratios['prec']['kk'] == min(self.precision_k_arr) / max(self.precision_k_arr)
        assert ratios['prec']['m'] == min(self.precision_m_arr) / max(self.precision_m_arr)

    def test_ratio_to_overall(self):
        self._prepare()

        ratios_overall = self.target.ratio(method='to_overall')
        assert isinstance(ratios_overall, pd.DataFrame)
        assert ratios_overall.shape == (2, 2)
        recall_k_overall = [x/self.recall_k for x in self.recall_k_arr] + \
            [self.recall_k/x for x in self.recall_k_arr]
        recall_m_overall = [x/self.recall_m for x in self.recall_m_arr] + \
            [self.recall_m/x for x in self.recall_m_arr]
        precision_k_overall = [x/self.precision_k for x in self.precision_k_arr] + \
            [self.precision_k/x for x in self.precision_k_arr]
        precision_m_overall = [x/self.precision_m for x in self.precision_m_arr] + \
            [self.precision_m/x for x in self.precision_m_arr]
        assert ratios_overall['recall']['kk'] == min(recall_k_overall)
        assert ratios_overall['recall']['m'] == min(recall_m_overall)
        assert ratios_overall['prec']['kk'] == min(precision_k_overall)
        assert ratios_overall['prec']['m'] == pytest.approx(min(precision_m_overall),
                                                            rel=1e-10, abs=1e-16)


def test_2m_1sf_2cf():
    # This test is structured differently, and hence not written as a class
    func_dict = {'recall': skm.recall_score, 'prec': skm.precision_score}
    target = metrics.MetricFrame(func_dict,
                                 y_t, y_p,
                                 sensitive_features=list(g_2),
                                 control_features={'cf0': g_3, 'cf1': g_1})

    assert isinstance(target.control_levels, list)
    assert (target.control_levels == ['cf0', 'cf1'])
    assert isinstance(target.sensitive_levels, list)
    assert (target.sensitive_levels == ['sensitive_feature_0'])

    # Check we have correct return types
    assert isinstance(target.overall, pd.DataFrame)
    assert isinstance(target.by_group, pd.DataFrame)

    mask_a = (g_1 == 'aa')
    mask_b = (g_1 == 'ba')
    mask_f = (g_2 == 'f')
    mask_g = (g_2 == 'g')
    mask_k = (g_3 == 'kk')
    mask_m = (g_3 == 'm')

    mask_k_a = np.logical_and(mask_k, mask_a)
    mask_k_b = np.logical_and(mask_k, mask_b)
    mask_m_a = np.logical_and(mask_m, mask_a)
    mask_m_b = np.logical_and(mask_m, mask_b)
    mask_k_a_f = np.logical_and(mask_k_a, mask_f)
    mask_k_a_g = np.logical_and(mask_k_a, mask_g)
    mask_k_b_f = np.logical_and(mask_k_b, mask_f)
    mask_k_b_g = np.logical_and(mask_k_b, mask_g)
    mask_m_a_f = np.logical_and(mask_m_a, mask_f)
    mask_m_a_g = np.logical_and(mask_m_a, mask_g)
    mask_m_b_f = np.logical_and(mask_m_b, mask_f)
    mask_m_b_g = np.logical_and(mask_m_b, mask_g)

    for mfn, metric_fn in func_dict.items():
        metric_k_a = metric_fn(y_t[mask_k_a], y_p[mask_k_a])
        metric_k_b = metric_fn(y_t[mask_k_b], y_p[mask_k_b])
        metric_m_a = metric_fn(y_t[mask_m_a], y_p[mask_m_a])
        metric_m_b = metric_fn(y_t[mask_m_b], y_p[mask_m_b])
        metric_k_a_f = metric_fn(y_t[mask_k_a_f], y_p[mask_k_a_f])
        metric_k_a_g = metric_fn(y_t[mask_k_a_g], y_p[mask_k_a_g])
        metric_k_b_f = metric_fn(y_t[mask_k_b_f], y_p[mask_k_b_f])
        metric_k_b_g = metric_fn(y_t[mask_k_b_g], y_p[mask_k_b_g])
        metric_m_a_f = metric_fn(y_t[mask_m_a_f], y_p[mask_m_a_f])
        metric_m_a_g = metric_fn(y_t[mask_m_a_g], y_p[mask_m_a_g])
        metric_m_b_f = metric_fn(y_t[mask_m_b_f], y_p[mask_m_b_f])
        metric_m_b_g = metric_fn(y_t[mask_m_b_g], y_p[mask_m_b_g])

        metric_k_a_arr = [metric_k_a_f, metric_k_a_g]
        metric_k_b_arr = [metric_k_b_f, metric_k_b_g]
        metric_m_a_arr = [metric_m_a_f, metric_m_a_g]
        metric_m_b_arr = [metric_m_b_f, metric_m_b_g]

        target_mins = target.group_min()
        assert isinstance(target_mins, pd.DataFrame)
        assert target_mins.shape == (4, 2)
        assert target_mins[mfn][('kk', 'aa')] == min(metric_k_a_arr)
        assert target_mins[mfn][('kk', 'ba')] == min(metric_k_b_arr)
        assert target_mins[mfn][('m', 'aa')] == min(metric_m_a_arr)
        assert target_mins[mfn][('m', 'ba')] == min(metric_m_b_arr)

        target_maxs = target.group_max()
        assert isinstance(target_mins, pd.DataFrame)
        assert target_maxs.shape == (4, 2)
        assert target_maxs[mfn][('kk', 'aa')] == max(metric_k_a_arr)
        assert target_maxs[mfn][('kk', 'ba')] == max(metric_k_b_arr)
        assert target_maxs[mfn][('m', 'aa')] == max(metric_m_a_arr)
        assert target_maxs[mfn][('m', 'ba')] == max(metric_m_b_arr)

        diffs = target.difference(method='between_groups')
        assert isinstance(diffs, pd.DataFrame)
        assert diffs.shape == (4, 2)
        assert diffs[mfn][('kk', 'aa')] == max(metric_k_a_arr) - min(metric_k_a_arr)
        assert diffs[mfn][('kk', 'ba')] == max(metric_k_b_arr) - min(metric_k_b_arr)
        assert diffs[mfn][('m', 'aa')] == max(metric_m_a_arr) - min(metric_m_a_arr)
        assert diffs[mfn][('m', 'ba')] == max(metric_m_b_arr) - min(metric_m_b_arr)

        diffs_overall = target.difference(method='to_overall')
        assert isinstance(diffs_overall, pd.DataFrame)
        assert diffs_overall.shape == (4, 2)
        diff_overall_k_a = max([abs(x-metric_k_a) for x in metric_k_a_arr])
        diff_overall_k_b = max([abs(x-metric_k_b) for x in metric_k_b_arr])
        diff_overall_m_a = max([abs(x-metric_m_a) for x in metric_m_a_arr])
        diff_overall_m_b = max([abs(x-metric_m_b) for x in metric_m_b_arr])

        assert diffs_overall[mfn][('kk', 'aa')] == diff_overall_k_a
        assert diffs_overall[mfn][('kk', 'ba')] == diff_overall_k_b
        assert diffs_overall[mfn][('m', 'aa')] == diff_overall_m_a
        assert diffs_overall[mfn][('m', 'ba')] == diff_overall_m_b

        ratios = target.ratio(method='between_groups')
        assert isinstance(ratios, pd.DataFrame)
        assert ratios.shape == (4, 2)
        assert ratios[mfn][('kk', 'aa')] == min(metric_k_a_arr) / max(metric_k_a_arr)
        assert ratios[mfn][('kk', 'ba')] == min(metric_k_b_arr) / max(metric_k_b_arr)
        assert ratios[mfn][('m', 'aa')] == min(metric_m_a_arr) / max(metric_m_a_arr)
        assert ratios[mfn][('m', 'ba')] == min(metric_m_b_arr) / max(metric_m_b_arr)

        ratios_overall = target.ratio(method='to_overall')
        assert isinstance(ratios_overall, pd.DataFrame)
        assert ratios_overall.shape == (4, 2)
        ratio_overall_k_a = [x/metric_k_a for x in metric_k_a_arr] + \
            [metric_k_a/x for x in metric_k_a_arr]
        ratio_overall_k_b = [x/metric_k_b for x in metric_k_b_arr] + \
            [metric_k_b/x for x in metric_k_b_arr]
        ratio_overall_m_a = [x/metric_m_a for x in metric_m_a_arr] + \
            [metric_m_a/x for x in metric_m_a_arr]
        ratio_overall_m_b = [x/metric_m_b for x in metric_m_b_arr] + \
            [metric_m_b/x for x in metric_m_b_arr]
        assert ratios_overall[mfn][('kk', 'aa')] == min(ratio_overall_k_a)
        assert ratios_overall[mfn][('kk', 'ba')] == min(ratio_overall_k_b)
        assert ratios_overall[mfn][('m', 'aa')] == min(ratio_overall_m_a)
        assert ratios_overall[mfn][('m', 'ba')] == pytest.approx(min(ratio_overall_m_b),
                                                                 rel=1e-10, abs=1e-16)
