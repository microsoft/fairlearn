from sklearn import __version__ as sklearn_version
from sklearn.utils.validation import check_array as _check_array


def validate_data(estimator, X, **kwargs):  # pragma: no cover
    """
    Validate data for sklearn estimators. This function provides compatibility across
    different sklearn versions by handling the validation method changes between versions.

    Parameters
    ----------
    estimator : object
        Sklearn estimator instance
    X : array-like
        Input data to validate
    **kwargs : dict
        Additional keyword arguments passed to the validation function

    Returns
    -------
    array-like
        The validated input data

    Notes
    -----
    For sklearn versions >= 1.6, uses estimator._validate_data()
    For sklearn versions < 1.6, uses sklearn.utils.validation.validate_data()
    """
    try:
        from sklearn.utils.validation import validate_data

        return validate_data(estimator, X, **kwargs)
    except ImportError:
        # sklearn version < 1.6
        ensure_all_finite = kwargs.pop("ensure_all_finite", True)

        return estimator._validate_data(X, **kwargs, force_all_finite=ensure_all_finite)


def check_array(X, **kwargs):  # pragma: no cover
    """
    Wrapper for sklearn's check_array function that handles version compatibility.
    Using this private function is only necessary if the call contains
    force_all_finite for sklearn version < 1.6 or ensure_all_finite for sklearn version >= 1.6.
    Parameters
    ----------
    X : array-like
        Input array to check/convert
    **kwargs : dict
        Additional arguments to pass to sklearn's check_array function

    Returns
    -------
    array-like
        The checked and potentially converted input array

    Notes
    -----
    This function provides compatibility across different sklearn versions by
    handling changes in the check_array function behavior.
    """
    try:
        return _check_array(X, **kwargs)
    except TypeError:
        # sklearn version < 1.6
        kwargs.pop("ensure_all_finite", False)
        return _check_array(X, **kwargs, force_all_finite=False)


# the two functions below were copied from sklearn-compat PR #15
# both functions are use to accommodate the usage of sklearn 1.6 and the versions below with one API
def patched_more_tags(estimator, expected_failed_checks):  # pragma: no cover
    """
    Patch an estimator's _more_tags method to include expected failed checks.
    This is used for compatibility with sklearn's estimator checks framework.

    Parameters
    ----------
    estimator : object
        The sklearn estimator instance to patch
    expected_failed_checks : list or callable
        List of check names that are expected to fail, or a callable that returns
        such a list when given an estimator instance

    Returns
    -------
    object
        The patched estimator with modified _more_tags method
    """
    import copy

    from sklearn.utils._tags import _safe_tags

    original_tags = copy.deepcopy(_safe_tags(estimator))

    def patched_more_tags(self):
        original_tags.update({"_xfail_checks": expected_failed_checks})
        return original_tags

    estimator.__class__._more_tags = patched_more_tags
    return estimator


def parametrize_with_checks(  # pragma: no cover
    estimators,
    *,
    legacy=True,
    expected_failed_checks=None,
):
    """
    Parametrize a test with a list of estimators and their expected failed checks.

    This function is a wrapper around sklearn's parametrize_with_checks, allowing
    for additional handling of expected failed checks for each estimator.

    Parameters
    ----------
    estimators : list
        A list of sklearn estimator instances to be tested.
    legacy : bool, optional
        This parameter is not supported and is ignored.
    expected_failed_checks : callable, optional
        A callable that returns a list of check names expected to fail for a given
        estimator instance.

    Returns
    -------
    function
        A pytest parameterized test function with the given estimators.
    """
    from sklearn.utils.estimator_checks import parametrize_with_checks

    if sklearn_version < "1.6":
        estimators = [
            patched_more_tags(estimator, expected_failed_checks(estimator))
            for estimator in estimators
        ]
        return parametrize_with_checks(estimators)
    else:
        return parametrize_with_checks(estimators, expected_failed_checks=expected_failed_checks)
