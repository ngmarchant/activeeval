import numpy as np
from typing import Union, Tuple, Optional
from numpy import ndarray

from ._base_proposals import BaseProposal
from ..pools import BasePool


class Passive(BaseProposal):
    """Passive proposal

    Proposes instances to label by sampling from the pool uniformly at random.

    Parameters
    ----------
    pool : an instance of activeeval.pool.BasePool
        A pool that specifies the support of the proposal.

    random_state : int, None or numpy.random.RandomState, optional (default = None)
        If int, random_state is the seed used by the random number
        generator; if RandomState instance, random_state is the random
        number generator; if None, the random number generator is the
        RandomState instance used by `numpy.random`.

    replace : bool, optional (default = True)
        Whether to draw instances from the pool with replacement.
    """
    def __init__(self, pool: BasePool, random_state: Union[int, float, np.random.RandomState, None] = None,
                 replace: bool = True) -> None:
        super().__init__(pool, random_state, replace)
        self._instance_weight = 1.0 / self.pool.n_instances
        self._not_sampled_idx = None if self.replace else set(range(self.pool.n_instances))

    def _draw_impl(self, size: Optional[int]) -> ndarray:
        if self.replace:
            return self.random_state.randint(0, self.pool.n_instances, size)
        else:
            idx = self.random_state.choice(tuple(self._not_sampled_idx), size, replace=False)
            for i in idx:
                self._not_sampled_idx.remove(i)
            return idx

    def get_pmf(self, idx: Union[ndarray, int, None] = None) -> Union[ndarray, float]:
        if np.isscalar(idx):
            if self.replace:
                return self._instance_weight
            else:
                if self._not_sampled_idx:
                    return (idx in self._not_sampled_idx) / len(self._not_sampled_idx)
                else:
                    return np.nan
        else:
            if self.replace:
                len_output = self.pool.n_instances if idx is None else len(idx)
                return np.repeat(self._instance_weight, len_output)
            else:
                if not self._not_sampled_idx:
                    return np.array([], dtype=float)
                if idx is None:
                    output = np.zeros(self.pool.n_instances, dtype=float)
                    output[tuple(self._not_sampled_idx)] = 1.0 / len(self._not_sampled_idx)
                    return output
                else:
                    output = np.array([(i in self._not_sampled_idx) for i in idx], dtype=float)
                    return output / len(self._not_sampled_idx)

    def get_weight(self, idx: Union[ndarray, int, None] = None) -> Union[ndarray, float]:
        if np.isscalar(idx):
            return 1.0
        else:
            len_output = self.pool.n_instances if idx is None else len(idx)
            return np.ones(len_output, dtype=np.float)

    def draw(self, size: Optional[int] = None, return_weight: bool = True) -> \
            Union[ndarray, int, Tuple[ndarray, ndarray], Tuple[int, float]]:
        return super().draw(size, return_weight)

    def reset(self) -> None:
        self._not_sampled_idx = None if self.replace else set(range(self.pool.n_instances))
