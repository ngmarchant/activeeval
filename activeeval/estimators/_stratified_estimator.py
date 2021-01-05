import warnings
import numpy as np
from typing import Union, Optional, Iterable
from numpy import ndarray
from scipy.sparse import spmatrix

from ..proposals import BaseProposal
from ..pools import BasePartitionedPool
from ..measures import BaseMeasure
from ._base import BaseEstimator


class StratifiedEstimator(BaseEstimator):
    """Stratified Estimator

    This estimator is suitable for use with stratified sampling schemes.
    It assumes instances are partitioned into strata, and that the sampling
    weights on instances are *uniform* within each stratum.

    Parameters
    ----------
    measure : an instance of activeeval.measures.BaseMeasure
        A target generalized measure to estimate.

    pool : an instance of activeeval.pool.BasePartitionedPool
        A partitioned pool of unlabeled instances.
    """
    def __init__(self, measure: BaseMeasure, pool: BasePartitionedPool) -> None:
        super().__init__(measure)
        if not issubclass(type(pool), BasePartitionedPool):
            raise TypeError("`pool` must be an instance of activeeval.pool.BasePartitionedPool")
        self.pool = pool
        # Overwrite from base class
        self._risk_numerator = np.zeros((pool.n_blocks, measure.n_dim_risk), dtype=float)
        self._n_samples = np.zeros(pool.n_blocks, dtype=int)

    def update(self, idx: Union[int, ndarray, Iterable], y: Union[int, float, ndarray, Iterable],
               weight: Union[float, ndarray, Iterable] = 1.0, x: Union[int, float, ndarray, None] = None,
               proposal: Optional[BaseProposal] = None) -> None:
        super().update(idx, y, weight, x, proposal)
        if not np.all(weight == 1.0):
            raise ValueError("`weight` must be 1.0 for this estimator")
        idx, y, weight = np.atleast_1d(idx, y, weight)
        x = np.atleast_1d(x) if x is not None else x
        loss = self.measure.loss(idx, y, x)
        if isinstance(loss, spmatrix):
            loss = loss.toarray()
        block_ids = self.pool.block_assignments[idx]
        np.add.at(self._risk_numerator, block_ids, loss)
        np.add.at(self._n_samples, block_ids, 1)
        self.n_samples += y.shape[0]

    def get(self) -> Union[ndarray, float]:
        if self.n_samples == 0:
            warnings.warn("cannot produce an estimate without any samples", UserWarning)
            return np.nan
        # Get ids of blocks which haven't been sampled yet
        sampled = self._n_samples > 0
        if (self.pool.n_blocks - sampled.sum()) > 0:
            # "Remove" the parts that haven't been sampled
            weights = self.pool.block_weights[sampled] / self.pool.block_weights[sampled].sum()
            risk = np.einsum('k,k,kd->d', weights, 1 / self._n_samples[sampled],
                             self._risk_numerator[sampled])
        else:
            risk = np.einsum('k,k,kd->d', self.pool.block_weights, 1 / self._n_samples, self._risk_numerator)
        measure = self.measure.g(risk)
        try:
            return measure.item()
        except ValueError:
            return measure

    def reset(self) -> None:
        super().reset()
        self._risk_numerator = np.zeros((self.pool.n_blocks, self.measure.n_dim_risk), dtype=float)
        self._n_samples = np.zeros(self.pool.n_blocks, dtype=int)
