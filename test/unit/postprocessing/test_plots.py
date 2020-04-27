# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pkg_resources
import platform
import pytest
import sys
from fairlearn.postprocessing import ThresholdOptimizer, plot_threshold_optimizer

from .conftest import scores_ex, ExamplePredictor, _data_ex1, _data_ex2, _data_ex3


PYTEST_MPL_NOT_INSTALLED_MSG = "skipping plotting tests because pytest-mpl is not installed"

"""Right now the baseline plot comparison doesn't succeed consistently on every
platform and is therefore disabled. To generate the baseline plots run the
following command from the root directory of the repository
python -m pytest test/unit/postprocessing/test_plots.py
    --mpl-generate-path=test/unit/postprocessing/baseline
Make sure to have `pytest-mpl` installed or this will not work.
pytest can run the tests either to check that there are no exceptions (using
a typical pytest command without extra options) or to actually compare the
generated images with the baseline plots (using pytest --mpl)."""


def _fit_and_plot(constraints, plotting_data):
    import matplotlib.pyplot as plt

    adjusted_predictor = ThresholdOptimizer(estimator=ExamplePredictor(scores_ex),
                                            constraints=constraints)
    adjusted_predictor.fit(plotting_data.X, plotting_data.y,
                           sensitive_features=plotting_data.sensitive_features)
    fig, (ax) = plt.subplots(1, 1)
    plot_threshold_optimizer(adjusted_predictor, ax=ax, show_plot=False)
    return fig


def is_mpl_installed():
    try:
        pkg_resources.get_distribution("pytest-mpl")
        pkg_resources.get_distribution("matplotlib")
        return True
    except pkg_resources.DistributionNotFound:
        return False


def is_py35_on_macos():
    return sys.version_info[0] == 3 and sys.version_info[1] == 5 and platform.system() == "Darwin"


@pytest.mark.skipif(is_py35_on_macos(), reason="Python 3.5 on MacOS requires TkAgg.")
@pytest.mark.skipif(not is_mpl_installed(), reason=PYTEST_MPL_NOT_INSTALLED_MSG)
class TestPlots:
    @pytest.mark.mpl_image_compare(filename="equalized_odds_ex1.png")
    def test_plot_equalized_odds_ex1(self):
        return _fit_and_plot('equalized_odds', _data_ex1)

    @pytest.mark.mpl_image_compare(filename="equalized_odds_ex2.png")
    def test_plot_equalized_odds_ex2(self):
        return _fit_and_plot('equalized_odds', _data_ex2)

    @pytest.mark.mpl_image_compare(filename="equalized_odds_ex3.png")
    def test_plot_equalized_odds_ex3(self):
        return _fit_and_plot('equalized_odds', _data_ex3)

    @pytest.mark.mpl_image_compare(filename="demographic_parity_ex1.png")
    def test_plot_demographic_parity_ex1(self):
        return _fit_and_plot('demographic_parity', _data_ex1)

    @pytest.mark.mpl_image_compare(filename="demographic_parity_ex2.png")
    def test_plot_demographic_parity_ex2(self):
        return _fit_and_plot('demographic_parity', _data_ex2)

    @pytest.mark.mpl_image_compare(filename="demographic_parity_ex3.png")
    def test_plot_demographic_parity_ex3(self):
        return _fit_and_plot('demographic_parity', _data_ex3)
