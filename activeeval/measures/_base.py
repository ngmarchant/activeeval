from abc import ABC
from typing import Union, Iterable
from numpy import ndarray
from scipy.sparse import spmatrix


class BaseMeasure(ABC):
    """An abstract base class representing a generalized measure

    A generalized measure is a real-valued vector that summarizes the
    performance of one or more *systems* under evaluation. Formally, it is
    defined in terms of a vector-valued *risk functional*

    .. math::

        R[\\ell, f] = \\mathbb{E}[\\ell(X, Y; f)],

    that takes on values in :math:`\\mathbb{R}^d`, where :math:`\\ell` is a
    loss function that depends on the input to the system(s) :math:`X`,
    the unknown true output :math:`Y`, plus the outputs from the system(s)
    under evaluation :math:`f` (a set of functions in general). The
    expectation is taken with respect to the joint distribution
    :math:`p(x, y) = \\frac{1}{N} \\sum_{i = 1}^{N} \\mathbb{I}[x = x_i] p(y|x_i)`
    on a pool of :math:`N` instances. Given the risk :math:`R`, the
    generalized measure :math:`G = g(R)` is obtained by applying a continuous
    mapping :math:`g: \\mathbb{R}^d \\to \\mathbb{R}^m` that is differentiable
    at :math:`R`.

    Attributes
    ----------
    n_dim_risk : int
        Dimension of the vector-valued risk (denoted :math:`d` above)

    n_dim_g : int
        Dimension of the range of the mapping function (denoted :math:`m`
        above)
    """
    def __init__(self) -> None:
        self.n_dim_risk = 0
        self.n_dim_g = 0

    def loss(self, idx: Union[int, ndarray, Iterable], y: Union[int, float, str, ndarray, Iterable],
             x: Union[int, float, ndarray, Iterable, None] = None) -> Union[ndarray, spmatrix]:
        """Evaluate the vector-valued loss function for a given set of
        instances. Outputs from the system(s) under evaluation do not need to
        be passed as arguments, as they are referenced internally within the
        class.

        Parameters
        ----------
        idx : array-like, shape (n_instances,)
            An array of integer ids referencing instances in the pool.

        y : array-like, shape (n_instances, n_dim_y)
            An array of outputs/labels corresponding to the instances
            referenced in idx. Each output may be a scalar or vector. The
            instances are indexed along the first axis.

        x : array-like, shape (n_instances, n_dim_x) or None, optional
            (default=None)
            An array of intputs/feature vectors corresponding to the instances
            referenced in idx. Each input may be a scalar or vector. The
            instances are indexed along the first axis.

        Returns
        -------
        numpy.ndarray or scipy.sparse.spmatrix, shape (n_instances, n_dim_risk)
            Loss function evaluated at the instances.
        """
        pass

    def _check_risk(self, risk: ndarray) -> None:
        if risk.ndim == 1:
            if risk.shape[0] != self.n_dim_risk:
                raise ValueError("`risk` must be a of length {}".format(self.n_dim_risk))
        else:
            raise ValueError("`risk` must be a 1d array")

    def g(self, risk: ndarray) -> ndarray:
        """Apply the mapping function at the given value of the risk.

        Parameters
        ----------
        risk : numpy.ndarray, shape (n_dim_risk,)
            Vector-valued risk.

        Returns
        -------
        numpy.ndarray, shape (n_dim_g,)
            Output of the mapping function.
        """
        pass

    def jacobian_g(self, risk: ndarray) -> Union[ndarray, spmatrix]:
        """Evaluate the Jacobian of the mapping function ``g`` with respect
        to the risk, at the given value of the risk.

        Parameters
        ----------
        risk : numpy.ndarray, shape (n_dim_risk,)
            Vector-valued risk.

        Returns
        -------
        numpy.ndarray or scipy.sparse.spmatrix, shape (n_dim_risk, n_dim_g)
            Jacobian of the mapping function.
        """
        pass
