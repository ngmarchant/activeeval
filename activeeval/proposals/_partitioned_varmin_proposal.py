import numpy as np
from typing import Union, Iterable, Optional
from numpy import ndarray
from scipy.sparse import spmatrix

from ._varmin_proposals import AdaptiveBaseProposal
from ._oracle_estimators import BasePartitionedOE
from ..pools import BasePartitionedPool
from ..measures import BaseMeasure


def compute_optimal_blocked_proposal(block_weights: ndarray, avg_loss_block: ndarray, avg_label_dist_block: ndarray,
                                     measure: BaseMeasure, normalized: bool = True) -> ndarray:
    """Compute the asymptotically optimal proposal for a given measure and
    pool

    Parameters
    ----------
    block_weights : numpy.ndarray, shape (n_blocks,)
        Weight of each block

    avg_loss_block : numpy.ndarray, shape (n_blocks, n_labels, measure.n_dim_risk)
        Average loss per block

    avg_label_dist_block : numpy.ndarray, shape (n_blocks, n_labels)
        Average ground truth label distribution p(y|block) per block.

    measure : an instance of activeeval.measures.BaseMeasure
        The target measure.

    normalized : bool, optional (default=True)
        Whether to normalize the probability mass function.

    Returns
    -------
    numpy.ndarray, shape (n_instances,)
        Probability mass function over the instances in the pool.
    """
    n_blocks, n_classes = avg_label_dist_block.shape
    block_weights = np.asarray(block_weights)
    avg_label_dist_block = np.asarray(avg_label_dist_block)
    if np.any((avg_label_dist_block < 0) | (avg_label_dist_block > 1)) or \
            not np.allclose(avg_label_dist_block.sum(axis=1), 1):
        raise ValueError("each row in `avg_label_dist_block` must be a normalized probability distribution")
    if not avg_loss_block.shape == (n_blocks, n_classes, measure.n_dim_risk):
        raise ValueError("`avg_loss_block` must have shape (n_blocks, n_classes, measure.n_dim_risk)")

    # Reshape to match 0th axis of loss
    avg_label_dist_block_weight = block_weights[:, np.newaxis] * avg_label_dist_block
    avg_loss_block = avg_loss_block.reshape(n_blocks * n_classes, measure.n_dim_risk)

    # Evaluate the risk. Works when loss is an ndarray or spmatrix.
    risk = avg_loss_block.T.dot(avg_label_dist_block_weight.ravel())

    jac_g = measure.jacobian_g(risk)

    # Implementation of multiplication with the loss depends on whether jac_g is an ndarray or spmatrix
    # Result has shape (measure.n_dim_g, pool.n_instances * n_classes)
    if isinstance(jac_g, spmatrix):
        # order below is more efficient if jac_g is a spmatrix
        loss_jac_g_sq = jac_g.dot(avg_loss_block.transpose())
    else:
        loss_jac_g_sq = avg_loss_block.dot(jac_g.transpose()).transpose()

    # Square the result, but implementation varies for spmatrix and ndarray
    if isinstance(loss_jac_g_sq, spmatrix):
        loss_jac_g_sq.data[:] = np.square(loss_jac_g_sq.data)
    else:
        loss_jac_g_sq = np.square(loss_jac_g_sq)

    # Sum over components of risk
    if isinstance(loss_jac_g_sq, spmatrix):
        # Result of sum() method is a np.matrix of shape (1, pool.n_instance * n_classes)
        # .A1 extracts the 1d ndarray.
        inner = loss_jac_g_sq.sum(axis=0).A1
    else:
        inner = np.add.reduce(loss_jac_g_sq, axis=0)

    # Take expectation with respect to label
    pmf = np.reshape(inner, (n_blocks, n_classes)) * avg_label_dist_block
    pmf = np.add.reduce(pmf, axis=1)
    pmf = np.sqrt(pmf)

    if normalized:
        pmf = pmf / pmf.sum()
    return pmf


class PartitionedAdaptiveVarMin(AdaptiveBaseProposal):
    """Adaptive stratified variance-minimizing proposal

    Proposes instances to label by sampling from an adaptive biased proposal
    distribution that seeks to minimize asymptotic variance. The biased
    proposal distribution is approximated using adaptive estimates of the
    oracle distribution :math:`p(y|x)` for each instance :math:`x` in the pool.

    Parameters
    ----------
    pool : an instance of activeeval.pool.BasePartitionedPool
        A partitioned pool that specifies the support of the proposal.

    measure : an instance of activeeval.measures.BaseMeasure
        The target measure. The proposal is selected so as to minimize
        the asymptotic variance for AIS estimates of this measure.

    oracle_estimator : an instance of activeeval.measure.BaseOE
        An adaptive estimator for the oracle distribution :math:`p(y|x)`

    epsilon : float, optional (default: 1e-9)
        A positive constant used for thresholding when approximating the
        optimal proposal. Smaller values result in a closer approximation.

    random_state : int, None or numpy.random.RandomState, optional
        (default = None)
        If int, random_state is the seed used by the random number generator;
        if RandomState instance, random_state is the random number generator;
        if None, the random number generator is the RandomState instance used by
        `numpy.random`.
    """
    def __init__(self, pool: BasePartitionedPool, measure: BaseMeasure, oracle_estimator: BasePartitionedOE,
                 epsilon: float = 1e-9, random_state: Union[int, float, np.random.RandomState, None] = None) -> None:
        super().__init__(pool, random_state=random_state)

        if not isinstance(measure, BaseMeasure):
            raise TypeError("`measure` must be an instance of activeeval.measures.BaseMeasure")
        self.measure = measure

        self.pool = pool
        if not isinstance(oracle_estimator, BasePartitionedOE):
            raise TypeError("`oracle_estimator` must be an instance of activeeval.proposals.BasePartitionedOE")
        self.oracle_estimator = oracle_estimator

        n_classes = self.oracle_estimator.labels.size
        self._avg_loss_block = self._compute_avg_loss_block(n_classes, pool, measure)

        if not (np.isscalar(epsilon) and epsilon > 0):
            raise ValueError("`epsilon` must be a positive float")
        else:
            self.epsilon = epsilon

        self._pmf = None
        self.update()

    @staticmethod
    def _compute_avg_loss_block(n_classes: int, pool: BasePartitionedPool, measure: BaseMeasure) -> ndarray:
        # Enumerate all (idx, y) combinations
        idx = np.repeat(np.arange(pool.n_instances), n_classes)
        y = np.tile(np.arange(n_classes), pool.n_instances)
        # Compute loss: 0th axis corresponds to (idx, y) combination. 1st axis corresponds to dimensions of the risk.
        loss = measure.loss(idx, y).reshape(pool.n_instances, n_classes * measure.n_dim_risk)
        if isinstance(loss, spmatrix):
            loss = loss.tocsr()
            avg_loss = [loss[block].mean(axis=0).A1 for block in pool.blocks_iter()]
        else:
            avg_loss = [loss[block].mean(axis=0) for block in pool.blocks_iter()]
        avg_loss = np.array(avg_loss)
        avg_loss = avg_loss.reshape(pool.n_blocks, n_classes, measure.n_dim_risk)
        return avg_loss

    def _draw_impl(self, size: Optional[int]) -> Union[ndarray, int]:
        block_ids = self.random_state.choice(self.pool.n_blocks, size, replace=True, p=self._pmf)
        instance_ids = [self.random_state.choice(self.pool.block(block_id)) for block_id in block_ids]
        instance_ids = np.array(instance_ids)
        if size is None:
            return instance_ids.item()
        else:
            return instance_ids

    def get_pmf(self, idx: Union[ndarray, int, None] = None) -> Union[ndarray, float]:
        block_id = self.pool.block_assignments if idx is None else self.pool.block_assignments[idx]
        return self._pmf[block_id] / self.pool.block_sizes[block_id]

    def get_weight(self, idx: Union[ndarray, int, None] = None) -> Union[ndarray, float]:
        # Switch off warnings since denominator may be zero
        with np.errstate(divide='ignore'):
            return 1.0 / (self.pool.n_instances * self.get_pmf(idx))

    def _update_impl(self, idx: ndarray, y: Optional[ndarray], weight: ndarray) -> None:
        x = self.pool[idx]
        self.oracle_estimator.update(idx=idx, x=x, y=y, weight=weight)

    def update(self, idx: Union[ndarray, int, Iterable, None] = None,
               y: Union[ndarray, int, Iterable, None] = None,
               weight: Union[ndarray, float, Iterable, None] = None) -> None:
        super().update(idx, y, weight)

        conditional = self.oracle_estimator.predict_block()
        self._pmf = compute_optimal_blocked_proposal(self.pool.block_weights, self._avg_loss_block, conditional,
                                                     self.measure)
        # Epsilon-greedy
        self._pmf = (1 - self.epsilon) * self._pmf + self.epsilon * self.pool.block_weights

    def reset(self) -> None:
        self.oracle_estimator.reset()
        self._pmf = None
        self.update()
