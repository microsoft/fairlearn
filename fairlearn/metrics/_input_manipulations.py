# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np


def _ensure_1d_ndarray(input, input_name):
    """Ensures that the input is represented as a 1d numpy.ndarray.

    The goal of this routine is to ensure that input arrays of shape
    (1,n) and (n,1), not to mention (1,1,n,1,1) can all be treated as
    an array of shape (n). However, an array of shape (2,2) will be
    rejected.

    This routine relies on the behaviour of numpy.asarray, and is not
    comprehensive as a result. For example
    numpy.asarray([[1,2], [2]])
    will result in a 1D ndarray, with two elements, each of which is a
    list. This method is not built to detect that issue
    """
    result = np.asarray(input)

    if result.size > 1:
        result = np.squeeze(result)
    else:
        result = result.reshape(1)

    return result
