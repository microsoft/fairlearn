# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pytest
import sklearn.metrics as skm

from fairlearn.metrics import MetricFrame, count

from .data_for_test import g_1, g_2, y_p, y_t

# Have fixtures so that tests can be specific without
# constantly recomputing the same bootstrapping

# Use regression metrics since they are a little better behaved
# when testing that median \approx nominal
# The ratios in many metrics can wreak havoc with bootstrapping
# due to skewed distributions

N_BOOTSTRAP = 100
QUANTILES = [0.05, 0.5, 0.95]
ERROR_OPTIONS = ["raise", "coerce"]

ABS_TOL = 0.1


@pytest.fixture(scope="session")
def mf_1m_0cf():
    target_boot = MetricFrame(
        metrics=skm.mean_squared_error,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
        n_boot=N_BOOTSTRAP,
        ci_quantiles=QUANTILES,
        random_state=13489623,
    )
    return target_boot


@pytest.fixture(scope="session")
def mf_1mdict_0cf():
    target_boot = MetricFrame(
        metrics={"mse": skm.mean_squared_error},
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
        n_boot=N_BOOTSTRAP,
        ci_quantiles=QUANTILES,
        random_state=13489623,
    )
    return target_boot


@pytest.fixture(scope="session")
def mf_1m_1cf():
    target_boot = MetricFrame(
        metrics=skm.mean_squared_error,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
        control_features=g_2,
        n_boot=N_BOOTSTRAP,
        ci_quantiles=QUANTILES,
        random_state=13489623,
    )
    return target_boot


@pytest.fixture(scope="session")
def mf_2m_1cf():
    metric_dict = {
        "mse": skm.mean_squared_error,
        "group_frac": lambda y_true, y_pred: count(y_true, y_pred) / len(y_t),
    }
    target_boot = MetricFrame(
        metrics=metric_dict,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
        control_features=g_2,
        n_boot=N_BOOTSTRAP,
        ci_quantiles=QUANTILES,
        random_state=13489623,
    )
    return target_boot


class TestOverallQuantiles:
    def test_1m_0cf(self, mf_1m_0cf: MetricFrame):
        assert isinstance(mf_1m_0cf.overall_ci, list)
        assert len(mf_1m_0cf.overall_ci) == len(QUANTILES)
        assert mf_1m_0cf.ci_quantiles == QUANTILES
        # Overall value should be close to quantile 0.5
        assert mf_1m_0cf.overall_ci[1] == pytest.approx(mf_1m_0cf.overall, abs=ABS_TOL)

    def test_1m_0cf_dict(self, mf_1mdict_0cf):
        assert isinstance(mf_1mdict_0cf.overall_ci, list)
        assert len(mf_1mdict_0cf.overall_ci) == len(QUANTILES)
        assert mf_1mdict_0cf.ci_quantiles == QUANTILES
        # Overall value should be close to quantile 0.5
        assert mf_1mdict_0cf.overall_ci[1]["mse"] == pytest.approx(
            mf_1mdict_0cf.overall["mse"], abs=ABS_TOL
        )

    def test_1m_1cf(self, mf_1m_1cf):
        assert isinstance(mf_1m_1cf.overall_ci, list)
        assert len(mf_1m_1cf.overall_ci) == len(QUANTILES)
        assert mf_1m_1cf.ci_quantiles == QUANTILES

        # Overall value should be close to quantile 0.5
        assert mf_1m_1cf.overall_ci[1]["f"] == pytest.approx(
            mf_1m_1cf.overall["f"], abs=ABS_TOL
        )
        assert mf_1m_1cf.overall_ci[1]["g"] == pytest.approx(
            mf_1m_1cf.overall["g"], abs=ABS_TOL
        )

    def test_2m_1cf(self, mf_2m_1cf):
        assert isinstance(mf_2m_1cf.overall_ci, list)
        assert len(mf_2m_1cf.overall_ci) == len(QUANTILES)
        assert mf_2m_1cf.ci_quantiles == QUANTILES

        # Overall value should be close to quantile 0.5
        for m in ["mse", "group_frac"]:
            for cf in np.unique(g_2):
                assert mf_2m_1cf.overall_ci[1][m][cf] == pytest.approx(
                    mf_2m_1cf.overall[m][cf], abs=ABS_TOL
                )


class TestByGroupQuantiles:
    def test_1m_0cf(self, mf_1m_0cf: MetricFrame):
        assert isinstance(mf_1m_0cf.by_group_ci, list)
        assert len(mf_1m_0cf.by_group_ci) == len(QUANTILES)
        assert mf_1m_0cf.ci_quantiles == QUANTILES
        for g in np.unique(g_1):
            # Check median close to nominal
            assert mf_1m_0cf.by_group_ci[1][g] == pytest.approx(
                mf_1m_0cf.by_group[g], abs=ABS_TOL
            )

    def test_1m_0cf_dict(self, mf_1mdict_0cf: MetricFrame):
        assert isinstance(mf_1mdict_0cf.by_group_ci, list)
        assert len(mf_1mdict_0cf.by_group_ci) == len(QUANTILES)
        assert mf_1mdict_0cf.ci_quantiles == QUANTILES
        for g in np.unique(g_1):
            # Check median close to nominal
            assert mf_1mdict_0cf.by_group_ci[1]["mse"][g] == pytest.approx(
                mf_1mdict_0cf.by_group["mse"][g], abs=ABS_TOL
            )

    def test_1m_1cf(self, mf_1m_1cf: MetricFrame):
        assert isinstance(mf_1m_1cf.by_group_ci, list)
        assert len(mf_1m_1cf.by_group_ci) == len(QUANTILES)
        assert mf_1m_1cf.ci_quantiles == QUANTILES
        for cf in np.unique(g_2):
            for g in np.unique(g_1):
                # Check median close to nominal
                assert mf_1m_1cf.by_group_ci[1][cf][g] == pytest.approx(
                    mf_1m_1cf.by_group[cf][g], abs=ABS_TOL
                )

    def test_2m_1cf(self, mf_2m_1cf: MetricFrame):
        assert isinstance(mf_2m_1cf.by_group_ci, list)
        assert len(mf_2m_1cf.by_group_ci) == len(QUANTILES)
        assert mf_2m_1cf.ci_quantiles == QUANTILES

        for m in ["mse", "group_frac"]:
            for cf in np.unique(g_2):
                for g in np.unique(g_1):
                    # Check median close to nominal
                    assert mf_2m_1cf.by_group_ci[1][m][cf][g] == pytest.approx(
                        mf_2m_1cf.by_group[m][cf][g], abs=ABS_TOL
                    )


class TestGroupExtremes:
    def _get_comparators(self, mf: MetricFrame, aggregation: str, error_handling: str):
        if aggregation == "min":
            actual = mf.group_min_ci(errors=error_handling)
            nominal = mf.group_min(errors=error_handling)
        elif aggregation == "max":
            actual = mf.group_max_ci(errors=error_handling)
            nominal = mf.group_max(errors=error_handling)
        else:
            raise ValueError(f"Unrecognised option: {aggregation}")

        return actual, nominal

    @pytest.mark.parametrize("aggregation", ["min", "max"])
    @pytest.mark.parametrize("error_handling", ERROR_OPTIONS)
    def test_1m_0cf(
        self, mf_1m_0cf: MetricFrame, aggregation: str, error_handling: str
    ):
        result, nominal = self._get_comparators(mf_1m_0cf, aggregation, error_handling)
        assert isinstance(result, list)
        assert len(result) == len(QUANTILES)
        assert mf_1m_0cf.ci_quantiles == QUANTILES
        # Check median close to nominal
        assert result[1] == pytest.approx(nominal, abs=ABS_TOL)

    @pytest.mark.parametrize("aggregation", ["min", "max"])
    @pytest.mark.parametrize("error_handling", ERROR_OPTIONS)
    def test_1m_0cf_dict(
        self, mf_1mdict_0cf: MetricFrame, aggregation: str, error_handling: str
    ):
        result, nominal = self._get_comparators(
            mf_1mdict_0cf, aggregation, error_handling
        )
        assert isinstance(result, list)
        assert len(result) == len(QUANTILES)
        assert mf_1mdict_0cf.ci_quantiles == QUANTILES
        # Check median close to nominal
        assert result[1]["mse"] == pytest.approx(nominal["mse"], abs=ABS_TOL)

    @pytest.mark.parametrize("aggregation", ["min", "max"])
    @pytest.mark.parametrize("error_handling", ERROR_OPTIONS)
    def test_1m_1cf(
        self, mf_1m_1cf: MetricFrame, aggregation: str, error_handling: str
    ):
        result, nominal = self._get_comparators(mf_1m_1cf, aggregation, error_handling)
        assert isinstance(result, list)
        assert len(result) == len(QUANTILES)
        assert mf_1m_1cf.ci_quantiles == QUANTILES
        for cf in np.unique(g_2):
            # Check median close to nominal
            assert result[1][cf] == pytest.approx(nominal[cf], abs=ABS_TOL)

    @pytest.mark.parametrize("aggregation", ["min", "max"])
    @pytest.mark.parametrize("error_handling", ERROR_OPTIONS)
    def test_2m_1cf(
        self, mf_2m_1cf: MetricFrame, aggregation: str, error_handling: str
    ):
        result, nominal = self._get_comparators(mf_2m_1cf, aggregation, error_handling)
        assert isinstance(result, list)
        assert len(result) == len(QUANTILES)
        assert mf_2m_1cf.ci_quantiles == QUANTILES

        for m in ["mse", "group_frac"]:
            for cf in np.unique(g_2):
                # Check median close to nominal
                assert result[1][m][cf] == pytest.approx(nominal[m][cf], abs=ABS_TOL)


class TestGroupComparisons:
    def _get_comparators(
        self, mf: MetricFrame, comparator: str, method: str, error_handling: str
    ):
        if comparator == "difference":
            actual = mf.difference_ci(method=method, errors=error_handling)
            nominal = mf.difference(method=method, errors=error_handling)
            tol = ABS_TOL
        elif comparator == "ratio":
            actual = mf.ratio_ci(method=method, errors=error_handling)
            nominal = mf.ratio(method=method, errors=error_handling)
            tol = 2 * ABS_TOL  # Ratios tend to be poorly behaved
        else:
            raise ValueError(f"Unrecognised option: {comparator}")

        return actual, nominal, tol

    @pytest.mark.parametrize("comparator", ["difference", "ratio"])
    @pytest.mark.parametrize("method", ["between_groups", "to_overall"])
    @pytest.mark.parametrize("error_handling", ERROR_OPTIONS)
    def test_1m_0cf(
        self, mf_1m_0cf: MetricFrame, comparator: str, error_handling: str, method: str
    ):
        result, nominal, _ = self._get_comparators(
            mf_1m_0cf, comparator, method, error_handling
        )
        assert isinstance(result, list)
        assert len(result) == len(QUANTILES)
        assert mf_1m_0cf.ci_quantiles == QUANTILES
        # Check median close to nominal
        print(f"result={result}")
        assert result[1] == pytest.approx(nominal, abs=ABS_TOL)

    @pytest.mark.parametrize("comparator", ["difference", "ratio"])
    @pytest.mark.parametrize("method", ["between_groups", "to_overall"])
    @pytest.mark.parametrize("error_handling", ERROR_OPTIONS)
    def test_1m_0cf_dict(
        self,
        mf_1mdict_0cf: MetricFrame,
        comparator: str,
        error_handling: str,
        method: str,
    ):
        result, nominal, _ = self._get_comparators(
            mf_1mdict_0cf, comparator, method, error_handling
        )
        assert isinstance(result, list)
        assert len(result) == len(QUANTILES)
        assert mf_1mdict_0cf.ci_quantiles == QUANTILES
        # Check median close to nominal
        assert result[1]["mse"] == pytest.approx(nominal["mse"], abs=ABS_TOL)

    @pytest.mark.parametrize("comparator", ["difference", "ratio"])
    @pytest.mark.parametrize("method", ["between_groups", "to_overall"])
    @pytest.mark.parametrize("error_handling", ERROR_OPTIONS)
    def test_1m_1cf(
        self,
        mf_1m_1cf: MetricFrame,
        comparator: str,
        error_handling: str,
        method: str,
    ):
        result, nominal, tol = self._get_comparators(
            mf_1m_1cf, comparator, method, error_handling
        )
        assert isinstance(result, list)
        assert len(result) == len(QUANTILES)
        assert mf_1m_1cf.ci_quantiles == QUANTILES
        for cf in np.unique(g_2):
            # Check median close to nominal
            assert result[1][cf] == pytest.approx(nominal[cf], abs=tol)

    @pytest.mark.parametrize("comparator", ["difference", "ratio"])
    @pytest.mark.parametrize("method", ["between_groups", "to_overall"])
    @pytest.mark.parametrize("error_handling", ERROR_OPTIONS)
    def test_2m_1cf(
        self,
        mf_2m_1cf: MetricFrame,
        comparator: str,
        error_handling: str,
        method: str,
    ):
        result, nominal, tol = self._get_comparators(
            mf_2m_1cf, comparator, method, error_handling
        )
        assert isinstance(result, list)
        assert len(result) == len(QUANTILES)
        assert mf_2m_1cf.ci_quantiles == QUANTILES

        for m in ["mse", "group_frac"]:
            for cf in np.unique(g_2):
                # Check median close to nominal
                assert result[1][m][cf] == pytest.approx(nominal[m][cf], abs=tol)
