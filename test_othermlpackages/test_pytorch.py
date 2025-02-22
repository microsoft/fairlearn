# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest
import sklearn
from packaging.version import parse

from fairlearn.adversarial import AdversarialFairnessClassifier

torch = pytest.importorskip("torch")
from skorch import NeuralNetClassifier  # noqa
from torch import nn, optim  # noqa

from fairlearn.reductions import DemographicParity  # noqa

from . import adversarial_fairness as af  # noqa
from . import package_test_common as ptc  # noqa


def create_model():
    # create model
    class ClassificationModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(103, 12),
                nn.ReLU(),
                nn.Linear(12, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Sigmoid(),
            )

        def forward(self, X, **kwargs):
            return self.model(X)

    class SampleWeightNeuralNet(NeuralNetClassifier):
        def __init__(self, *args, criterion__reduction="none", **kwargs):
            # make sure to set reduce=False in your criterion, since we need the loss
            # for each sample so that it can be weighted
            super().__init__(*args, criterion__reduction=criterion__reduction, **kwargs)

        def fit(self, X, y, sample_weight=None):
            if isinstance(X, (pd.DataFrame, pd.Series)):
                X = X.to_numpy().astype("float32")
            if isinstance(y, (pd.DataFrame, pd.Series)):
                y = y.to_numpy()
            if sample_weight is not None and isinstance(sample_weight, (pd.DataFrame, pd.Series)):
                sample_weight = sample_weight.to_numpy()
            y = y.reshape([-1, 1])

            sample_weight = sample_weight if sample_weight is not None else np.ones_like(y)
            X = {"X": X, "sample_weight": sample_weight}
            return super().fit(X, y)

        def predict(self, X):
            if isinstance(X, (pd.DataFrame, pd.Series)):
                X = X.to_numpy().astype("float32")
            return super().predict(X).squeeze()

        def get_loss(self, y_pred, y_true, X, *args, **kwargs):
            # override get_loss to use the sample_weight from X
            loss_unreduced = super().get_loss(y_pred, y_true.float(), X, *args, **kwargs)
            sample_weight = X["sample_weight"]
            sample_weight = sample_weight.to(loss_unreduced.device).unsqueeze(-1)
            # Need to put the sample weights on GPU
            loss_reduced = (sample_weight * loss_unreduced).mean()
            return loss_reduced

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = SampleWeightNeuralNet(
        ClassificationModel,
        max_epochs=10,
        optimizer=optim.Adam,
        lr=0.001,
        batch_size=512,
        # No validation
        train_split=None,
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
        criterion=nn.BCELoss,
        device=device,
    )
    return net


def _should_skip_test():
    return parse(sklearn.__version__) >= parse("1.6.0")


def test_examples():
    af.test_examples()


@pytest.mark.skipif(
    _should_skip_test(),
    reason="Skipped because of scikit-learn >= 1.6. Will be enabled again when the issues in the external library are fixed.",
)
def test_expgrad_classification():
    estimator = create_model()
    disparity_moment = DemographicParity()

    ptc.run_expgrad_classification(estimator, disparity_moment)


@pytest.mark.skipif(
    _should_skip_test(),
    reason="Skipped because of scikit-learn >= 1.6. Will be enabled again when the issues in the external library are fixed.",
)
def test_gridsearch_classification():
    estimator = create_model()
    disparity_moment = DemographicParity()

    ptc.run_gridsearch_classification(estimator, disparity_moment)


@pytest.mark.skipif(
    _should_skip_test(),
    reason="Skipped because of scikit-learn >= 1.6. Will be enabled again when the issues in the external library are fixed.",
)
def test_thresholdoptimizer_classification():
    estimator = create_model()

    ptc.run_thresholdoptimizer_classification(estimator)


def test_adversarial_classification():
    mitigator = AdversarialFairnessClassifier(
        backend="torch",
        predictor_model=[50, "relu"],
        adversary_model=[3, "relu"],
        random_state=123,
    )

    ptc.run_AdversarialFairness_classification(mitigator)
