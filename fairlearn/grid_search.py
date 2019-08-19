# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
This module implements a 1d grid search for binary classification with a single
binary protected attribute.

This is a simplification of the full exponentiated gradient search
and produces a sequence of models, each of which corresponds to a
different accuracy/fairness trade-off. The dimensionality of the search
increases with the number of possible values for the protected attribute,
so this technique does not scale beyond a binary protected attribute.
"""

import copy
import numpy as np

def _generate_p0_p1(labels):
    """ Function to compute p0 and p1 for the given
    set of labels
    """
    unique, counts = np.unique(labels, return_counts=True)

    if len(unique) != 2:
        raise RuntimeError("Supplied Y not binary")

    if not np.array_equal(unique, [0, 1]):
        raise RuntimeError("Supplied Y labels not 0 or 1")

    p0 = counts[0] / len(labels)
    p1 = counts[1] / len(labels)

    return p0, p1

def _generate_weights(labels, protected_attribute, L, p_ratio):
    weights = []

    for y, a in zip(labels, protected_attribute):
        w = 1e128
        if a == 0:
            w = 2*y - 1 - L * p_ratio
        else:
            w = 2*y - 1 + L

        weights.append(w)

    return np.array(weights)

def classification_binary_protected_1d(learner, x, y, protected_attribute, Ls=None, num_Ls=11):
    """Function to generate a list of models for a classification problem with
    a single binary protected attribute. The models sweep through different potential
    Lagrangian multipliers for the constrained optimisation problem (the constraint
    being demographic parity), each one corresponding to a particular tradeoff between
    fairness and accuracy

    :param learner: An object which can be used to fit a model to features, labels and
    weights. A deep copy of this is made for each value of the Lagrangian multiplier used
    :type learner: Must implement a fit(x, y, sample_weight) method
    :param x: The array of training data features (which may or may not contain the
    protected attribute). Must have as many rows as Y and A
    :type x: Numpy array with two dimensions or pandas dataframe
    :param y: The list of binary classes, which must be 0 or 1. Must contain the same
    number of entries as rows in X
    :type y: Numpy array with one dimension
    :param protected_attribute: The protected attribute corresponding to each row of X. Must be either 0 or 1
    :type protected_attribute: Numpy array with one dimension
    :param Ls: User specified set of Lagrangian multipliers to use for the optimisation
    problem. If this is set then num_Ls must be None. The result array will be equal in
    length to this array
    :type Ls: List of real numbers
    :param num_Ls: Specifies the number of Lagrangian multipliers to use in the
    optimisation problem. If this is set then Ls must be None. The result array will have
    as many entries as specified here

    :return: The models corresponding to each value of the Lagrangian multiplier tested
    :rtype: List of dictionaries. Each dictionary has fields "lambda" and "model." Each
    model will correspond to the input parameter learner after calling 'fit' on it (a deep
    copy is made). The user is responsible for converting these objects to an actual model,
    if further processing is required.
    """
    # Must specify either Ls or num_Ls
    if not (Ls is None) ^ (num_Ls is None):
        raise RuntimeError("Must specify either Ls or num_Ls")

    # Check that x, y and protected_attribute have the same number of rows
    # Check that y and protected_attribute are both 1d vectors
    if len(y.shape) != 1:
        raise RuntimeError("Supplied y not 1d vector")

    if len(protected_attribute.shape) != 1:
        raise RuntimeError("Supplied protected_attribute not 1d vector")

    if protected_attribute.shape[0] != y.shape[0]:
        raise RuntimeError("Supplied protected_attribute and y not same length")

    if x.shape[0] != y.shape[0]:
        raise RuntimeError("Supplied x and y do not have same number of rows")

    # Check that A only has values 0 and 1
    unique_protected_attribute_values = np.unique(protected_attribute)
    if not np.array_equal(unique_protected_attribute_values, [0, 1]):
        raise RuntimeError("Supplied protected_attribute labels not 0 or 1")

    # Compute p0 and p1
    # This will also check that Y is binary with values
    # 0 and 1
    p0, p1 = _generate_p0_p1(y)

    # If not supplied, generate array of trial Lagrangian multipliers
    if Ls is None:
        limit = 1
        if p0/p1 > 1:
            limit = p0/p1
        Ls = np.linspace(-2*limit, 2*limit, num_Ls)

    result = []
    for L in Ls:
        # Generate weights array
        sample_weights = _generate_weights(y, protected_attribute, L, p1/p0)

        # Generate Y'
        f = lambda x: 1 if x > 0 else 0
        re_labels = np.vectorize(f)(sample_weights)

        # Run the learner
        mylearner = copy.deepcopy(learner)
        mylearner.fit(x, re_labels, sample_weight=np.absolute(sample_weights))

        # Append the new learner, along with its L value to the result
        result.append({"model": mylearner, "lambda":L})

    # Return the result array (tuples of (L,model))
    return result
