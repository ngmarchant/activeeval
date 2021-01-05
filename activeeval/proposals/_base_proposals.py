from abc import ABC, abstractmethod
import numpy as np
from typing import Union, Tuple, Iterable, Optional
from numpy import ndarray

from ..pools import BasePool


class BaseProposal(ABC):
    """Base class for a proposal distribution

    Parameters
    ----------
    pool : an instance of activeeval.pool.BasePool
        A pool that specifies the support of the proposal.

    random_state : int, None or numpy.random.RandomState, optional
        (default = None)
        If int, random_state is the seed used by the random number generator;
        if RandomState instance, random_state is the random number generator;
        if None, the random number generator is the RandomState instance used
        by `numpy.random`.

    replace : bool, optional (default = True)
        Whether to draw items from the pool with replacement.
    """
    def __init__(self, pool: BasePool, random_state: Union[int, float, np.random.RandomState, None] = None,
                 replace: bool = True) -> None:
        if not isinstance(pool, BasePool):
            raise TypeError("`pool` must be an instance of activeeval.pool.BasePool")
        self.pool = pool

        if isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(random_state)

        if not isinstance(replace, (bool, np.bool)):
            raise TypeError("`replace` must be a bool")
        self.replace = replace

    @abstractmethod
    def get_pmf(self, idx: Union[ndarray, int, None] = None) -> Union[ndarray, float]:
        """Retrieve the proposal probability mass for the specified items

        Parameters
        ----------
        idx : int, numpy.ndarray or None, shape (n_items,), optional
            (default=None)
            Identifiers of items for which the probability mass is required.
            If None, returns the probability mass for all items in the pool.

        Returns
        -------
        pmf : numpy.ndarray, shape (n_items,)
            The probability mass for the specified items.
        """
        pass

    @abstractmethod
    def get_weight(self, idx: Union[ndarray, int, None] = None) -> Union[ndarray, float]:
        """Retrieve the importance weights for the specified items

        The importance weight is a ratio of the uniform probability mass and
        the proposal probability mass and is defined as::

            weight[idx] := 1/(n_items * pmf[idx])

        Parameters
        ----------
        idx : int, numpy.ndarray or None, shape (n_items,), optional
            (default=None)
            Identifiers of items for which the importance weight is required.
            If None, returns the importance weight for all items in the pool.

        Returns
        -------
        pmf : numpy.ndarray, shape (n_items,)
            The importance weights for the specified items
        """
        pass

    @abstractmethod
    def _draw_impl(self, size: Optional[int]) -> ndarray:
        """Class-specific implementation of draw
        """
        pass

    def draw(self, size: Union[int, None] = None, return_weight: bool = True) -> \
            Union[ndarray, int, Tuple[ndarray, ndarray], Tuple[int, float]]:
        """Draw items according to the proposal

        Parameters
        ----------
        size : int or None, optional (default=None)
            Number of draws from the proposal. If None, returns a single
            instance.

        return_weight: bool, optional (default=True)
            Whether to return the importance weights associated with each
            drawn instance.

        Returns
        -------
        If `return_weight` is False:
        idx : numpy.ndarray, shape (size,)
            Identifiers of the drawn items

        If `return_weight` is True:
        idx : numpy.ndarray, shape (size,)
            Identifiers of the drawn items
        weights : numpy.ndarray, shape (size,)
            Importance weights of the drawn items
        """
        idx = self._draw_impl(size)
        if return_weight:
            return idx, self.get_weight(idx)
        else:
            return idx

    def reset(self) -> None:
        """Reset the internal state of the proposal
        """
        pass


class AdaptiveBaseProposal(BaseProposal):
    """Base class for an adaptive proposal distribution

    Parameters
    ----------
    pool : an instance of activeeval.pool.BasePool
        A pool that specifies the support of the proposal.

    random_state : int, None or numpy.random.RandomState, optional
        (default = None)
        If int, random_state is the seed used by the random number
        generator; if RandomState instance, random_state is the random
        number generator; if None, the random number generator is the
        RandomState instance used by `numpy.random`.

    replace : bool, optional (default = True)
        Whether to draw items from the pool with replacement.
    """
    def __init__(self, pool: BasePool, random_state: Union[int, float, np.random.RandomState, None] = None,
                 replace: bool = True) -> None:
        super().__init__(pool, random_state, replace)

    @abstractmethod
    def _update_impl(self, idx: ndarray, y: Optional[ndarray], weight: ndarray) -> None:
        """Class-specific implementation of the update method
        """
        pass

    def update(self, idx: Union[ndarray, int, Iterable, None] = None,
               y: Union[ndarray, int, str, Iterable, None] = None,
               weight: Union[ndarray, float, Iterable, None] = None) -> None:
        """Update the internal state of the proposal. The update may or may
        not be based on observed data.

        Parameters
        ----------
        idx : int array-like or None, optional (default=None)
            Identifier(s) of observed instance(s).

        y : int, str, array-like or None, optional (default=None)
            Label(s) of the observed instance(s).

        weight: float, array-like or None, optional (default=None)
            Importance weight(s) associated with the observed instance(s).
        """
        if (idx is None) and (y is not None or weight is not None):
            raise ValueError("`idx` is not optional if `y` or `weight` is specified")
        if idx is not None:
            idx = np.atleast_1d(idx)
            if weight is None:
                weight = np.ones_like(idx, dtype=float)
            else:
                weight = np.atleast_1d(weight)
                if idx.shape != weight.shape:
                    raise ValueError("mismatch in shape of `idx` and `weight`")

            if y is not None:
                y = np.atleast_1d(y)
                if idx.shape != y.shape:
                    raise ValueError("mismatch in shape of `idx` and `y`")

            self._update_impl(idx, y, weight)

    @abstractmethod
    def reset(self) -> None:
        """Reset the internal state of the proposal
        """
        pass
