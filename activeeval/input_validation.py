import numpy as np
from numpy import ndarray


def check_label_input(y, name):
    if isinstance(y, np.ndarray):
        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError("{} array must have integer dtype".format(name))
        if y.ndim != 1:
            raise ValueError("{} array must be 1-dimensional".format(name))
    else:
        if not isinstance(y, (int, np.integer)):
            raise ValueError("{} must be a scalar integer or integer numpy.ndarray".format(name))


def check_input(y_true, y_pred):
    check_label_input(y_true, 'y_true')
    check_label_input(y_pred, 'y_pred')
    if isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray):
        if y_true.shape != y_pred.shape:
            raise ValueError("mismatch in length of y_true and y_pred")


def coerce_to_1d(x) -> ndarray:
    """Coerce input to a 1d array if possible, otherwise raise an error.
    """
    x = np.asarray(x)
    shape = np.shape(x)
    if x.ndim <= 1:
        return np.atleast_1d(x)
    if x.ndim == 2 and shape[1] == 1:
        return np.ravel(x)

    raise ValueError("Input with shape {} cannot be coerced to a 1d array".format(shape))


def assert_finite(x: ndarray):
    """Ensure the input array is finite
    """
    x = np.asanyarray(x)
    if not (x.dtype.kind in 'f'):
        raise ValueError("Input dtype {} is not a float".format(x.dtype))
    if not np.isfinite(x).all():
        raise ValueError("Input contains non-finite values")
