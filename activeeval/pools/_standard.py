import numpy as np
from typing import Union, Iterable, Optional
from numpy import ndarray

from ._base import BasePool


class Pool(BasePool):
    """A pool of unlabeled instances

    Parameters
    ----------
    n_instances : int, optional (default=None)
        Number of instances in the pool. Must be specified if `features` is
        None.

    features : array-like, shape (n_instances, n_features), optional (default=None)
        Feature vectors for instances in the pool.
    """
    def __init__(self, n_instances: Optional[int] = None, features: Union[ndarray, Iterable, None] = None) -> None:
        super().__init__()
        if n_instances is None and features is None:
            raise RuntimeError("at least one of `n_instances` or `features` must be specified")

        if n_instances is None:
            self._n_instances = features.shape[0]
        else:
            self._n_instances = n_instances
            if n_instances < 0:
                raise ValueError("`n_instances` must be a non-negative integer")

        self._features = None
        if features is not None:
            self._features = np.asarray(features)
            if self._features.ndim != 2:
                raise ValueError("`features` must be a 2-dimensional array")
            if self._features.shape[0] != self.n_instances:
                raise ValueError("`features` must have shape {} along the zero-th axis".format(self._n_instances))

    @property
    def n_instances(self) -> int:
        return self._n_instances

    def __getitem__(self, instance_ids: Union[int, ndarray, Iterable]) -> Optional[ndarray]:
        if self._features is not None:
            return self._features[instance_ids]
        else:
            return None
