# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
This module implements a 1d grid search for binary classification with a single
binary protected attribute.

This is a simplification of the full exponentiated gradient search
and produces a sequence of models, each of which corresponds to a
different accuracy/fairness trade-off
"""

import numpy as np
import pandas as pd
from collections import namedtuple

def _generate_p0_p1(Y):
    """ Function to compute p0 and p1 for the given
    set of labels Y
    """
    unique, counts = np.unique(Y, return_counts=True)

    
    return 0,1

def grid_search_classification_binary_protected_1d(learner, X, Y, A):
    """Function to generate a 1d set of models for a classification problem with
    a single binary protected attribute

    """

    # Check that X, Y and A have the same number of rows

    # Check that Y and A are both 1d vectors

    # Check that Y only has values 0 and 1

    # Check that A only has values 0 and 1

    # If needed, check Y only has values 0 and 1
    
    # Compute p0 and p1

    # If not supplied, generate array of trial Lagrangian multipliers

    for L in Ls:
        # Generate W array

        # Generate Y'

        # Run the learner
        model = learner.fit(abs(W), X, Yprime)

        # Append the new learner, along with its L value to the result

    # Return the result array (tuples of (L,model))

