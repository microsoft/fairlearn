# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest
import sklearn.metrics as skm

from .data_for_test import y_t, y_p, s_w, g_1, g_2, g_3, g_4
from .utils import _get_raw_MetricFrame


metric = [skm.recall_score,
          skm.precision_score,
          skm.accuracy_score,
          skm.balanced_accuracy_score]


@pytest.mark.parametrize("metric_fn", metric)
def test_1m_1sf_0cf(metric_fn):
    target = _get_raw_MetricFrame()
    func_dict = target._process_functions(metric_fn, None)
    assert target._user_supplied_callable is True
    sf_list = target._process_features("SF", g_1, y_t)

    result = target._compute_by_group(func_dict, y_t, y_p, sf_list, None)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 1)
    assert np.array_equal(result.index.names, ['SF0'])
    mask_a = (g_1 == 'aa')
    mask_b = (g_1 == 'ba')
    metric_a = metric_fn(y_t[mask_a], y_p[mask_a])
    metric_b = metric_fn(y_t[mask_b], y_p[mask_b])
    assert result[metric_fn.__name__]['aa'] == metric_a
    assert result[metric_fn.__name__]['ba'] == metric_b


@pytest.mark.parametrize("metric_fn", metric)
def test_1m_1sf_0cf_metric_dict(metric_fn):
    target = _get_raw_MetricFrame()
    func_dict = target._process_functions({metric_fn.__name__: metric_fn}, None)
    assert target._user_supplied_callable is False
    sf_list = target._process_features("SF", g_1, y_t)

    result = target._compute_by_group(func_dict, y_t, y_p, sf_list, None)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 1)
    assert np.array_equal(result.index.names, ['SF0'])
    mask_a = (g_1 == 'aa')
    mask_b = (g_1 == 'ba')
    metric_a = metric_fn(y_t[mask_a], y_p[mask_a])
    metric_b = metric_fn(y_t[mask_b], y_p[mask_b])
    assert result[metric_fn.__name__]['aa'] == metric_a
    assert result[metric_fn.__name__]['ba'] == metric_b


@ pytest.mark.parametrize("metric_fn", metric)
def test_1m_1sf_1cf(metric_fn):
    target = _get_raw_MetricFrame()
    func_dict = target._process_functions(metric_fn, None)
    assert target._user_supplied_callable is True
    sf_list = target._process_features("SF", g_1, y_t)
    cf_list = target._process_features("CF", g_2, y_t)

    result = target._compute_by_group(func_dict, y_t, y_p, sf_list, cf_list)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (4, 1)
    assert np.array_equal(result.index.names, ['CF0', 'SF0'])

    mask_a_f = np.logical_and((g_1 == 'aa'), (g_2 == 'f'))
    mask_a_g = np.logical_and((g_1 == 'aa'), (g_2 == 'g'))
    mask_b_f = np.logical_and((g_1 == 'ba'), (g_2 == 'f'))
    mask_b_g = np.logical_and((g_1 == 'ba'), (g_2 == 'g'))

    exp_a_f = metric_fn(y_t[mask_a_f], y_p[mask_a_f])
    exp_a_g = metric_fn(y_t[mask_a_g], y_p[mask_a_g])
    exp_b_f = metric_fn(y_t[mask_b_f], y_p[mask_b_f])
    exp_b_g = metric_fn(y_t[mask_b_g], y_p[mask_b_g])
    assert result[metric_fn.__name__][('f', 'aa')] == exp_a_f
    assert result[metric_fn.__name__][('f', 'ba')] == exp_b_f
    assert result[metric_fn.__name__][('g', 'aa')] == exp_a_g
    assert result[metric_fn.__name__][('g', 'ba')] == exp_b_g


@ pytest.mark.parametrize("metric_fn", metric)
def test_1m_1sf_1cf_metric_dict(metric_fn):
    target = _get_raw_MetricFrame()
    func_dict = target._process_functions({metric_fn.__name__: metric_fn}, None)
    assert target._user_supplied_callable is False
    sf_list = target._process_features("SF", g_1, y_t)
    cf_list = target._process_features("CF", g_2, y_t)

    result = target._compute_by_group(func_dict, y_t, y_p, sf_list, cf_list)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (4, 1)
    assert np.array_equal(result.index.names, ['CF0', 'SF0'])

    mask_a_f = np.logical_and((g_1 == 'aa'), (g_2 == 'f'))
    mask_a_g = np.logical_and((g_1 == 'aa'), (g_2 == 'g'))
    mask_b_f = np.logical_and((g_1 == 'ba'), (g_2 == 'f'))
    mask_b_g = np.logical_and((g_1 == 'ba'), (g_2 == 'g'))

    exp_a_f = metric_fn(y_t[mask_a_f], y_p[mask_a_f])
    exp_a_g = metric_fn(y_t[mask_a_g], y_p[mask_a_g])
    exp_b_f = metric_fn(y_t[mask_b_f], y_p[mask_b_f])
    exp_b_g = metric_fn(y_t[mask_b_g], y_p[mask_b_g])
    assert result[metric_fn.__name__][('f', 'aa')] == exp_a_f
    assert result[metric_fn.__name__][('f', 'ba')] == exp_b_f
    assert result[metric_fn.__name__][('g', 'aa')] == exp_a_g
    assert result[metric_fn.__name__][('g', 'ba')] == exp_b_g


def test_2m_2sf_2cf():
    target = _get_raw_MetricFrame()
    funcs = {'recall': skm.recall_score, 'prec': skm.precision_score}
    func_container_dict = target._process_functions(funcs,
                                                    {'recall': {'sample_weight': s_w}})
    sf_list = target._process_features("Sens", np.stack([g_1, g_3], axis=1), y_t)
    cf_list = target._process_features("Cond", {'cf1': g_2, 'cf2': g_4}, y_t)

    result = target._compute_by_group(func_container_dict, y_t, y_p, sf_list, cf_list)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (16, 2)
    assert np.array_equal(result.index.names, ['cf1', 'cf2', 'Sens0', 'Sens1'])

    # Only check some isolated results, rather than all 32
    mask_a_f = np.logical_and((g_1 == 'aa'), (g_2 == 'f'))
    mask_b_g = np.logical_and((g_1 == 'ba'), (g_2 == 'g'))
    mask_k_q = np.logical_and((g_3 == 'kk'), (g_4 == 'q'))

    mask_f_q_a_k = np.logical_and(mask_a_f, mask_k_q)
    recall_f_q_a_k = skm.recall_score(y_t[mask_f_q_a_k], y_p[mask_f_q_a_k],
                                      sample_weight=s_w[mask_f_q_a_k])
    prec_f_q_a_k = skm.precision_score(y_t[mask_f_q_a_k], y_p[mask_f_q_a_k])
    assert result['recall'][('f', 'q', 'aa', 'kk')] == recall_f_q_a_k
    assert result['prec'][('f', 'q', 'aa', 'kk')] == prec_f_q_a_k

    mask_g_q_b_k = np.logical_and(mask_b_g, mask_k_q)
    recall_g_q_b_k = skm.recall_score(y_t[mask_g_q_b_k], y_p[mask_g_q_b_k],
                                      sample_weight=s_w[mask_g_q_b_k])
    prec_g_q_b_k = skm.precision_score(y_t[mask_g_q_b_k], y_p[mask_g_q_b_k])
    assert result['recall'][('g', 'q', 'ba', 'kk')] == recall_g_q_b_k
    assert result['prec'][('g', 'q', 'ba', 'kk')] == prec_g_q_b_k
