import numpy as np
from typing import Union, Iterable, Optional
from numpy import ndarray
from scipy.sparse import spmatrix

from ._base_proposals import AdaptiveBaseProposal
from ._passive_proposal import Passive
from ._oracle_estimators import BaseOE
from ..pools import BasePool
from ..measures import BaseMeasure


def compute_optimal_proposal(pool: BasePool, labels: ndarray, true_label_dist: ndarray, measure: BaseMeasure,
                             epsilon: float = 0.0, normalized: bool = True, deterministic: bool = False) -> ndarray:
    """Compute the asymptotically optimal proposal for a given measure and
    pool

    Parameters
    ----------
    pool : an instance of activeeval.pool.BasePool
        A pool which specifies the support of the proposal.

    labels : numpy.ndarray, shape (n_classes,)
        Specifies the set of class labels, i.e. the support of the oracle
        response :math:`p(y|x)`.

    true_label_dist : numpy.ndarray, shape (n_instances, n_classes)
        Ground truth label distribution p(y|x) for each instance x in the pool.

    measure : an instance of activeeval.measures.BaseMeasure
        The target measure.

    epsilon : float, optional (default: 0.0)
        Non-negative threshold.

    normalized : bool, optional (default=True)
        Whether to normalize the probability mass function.

    deterministic : bool, optional (default=False)
        Whether the oracle response is deterministic.

    Returns
    -------
    numpy.ndarray, shape (n_instances,)
        Probability mass function over the instances in the pool.
    """
    if epsilon < 0:
        raise ValueError("`epsilon` must be a non-negative float")
    n_instances, n_classes = np.shape(true_label_dist)
    if n_instances != pool.n_instances:
        raise ValueError("length of zero-th dimension of `true_label_dist` must equal `pool.n_instances`")
    if n_classes != labels.size:
        raise ValueError("length of first dimension of `true_label_dist` must equal `labels.size`")
    if np.any((true_label_dist < 0) | (true_label_dist > 1)) or not np.allclose(true_label_dist.sum(axis=1), 1):
        raise ValueError("each row in `true_label_dist` must be a normalized probability distribution")

    # Enumerate all (idx, y) combinations
    idx = np.repeat(np.arange(n_instances), n_classes)
    y = np.tile(labels, n_instances)
    # Compute loss: 0th axis corresponds to (idx, y) combination. 1st axis corresponds to dimensions of the risk.
    # Later can be reshaped to (n_instances, n_classes, measure.n_dim_risk)
    loss = measure.loss(idx, y)

    # Reshape to match 0th axis of loss
    true_label_dist = true_label_dist.ravel()

    # Evaluate the risk. Works when loss is an ndarray or spmatrix.
    risk = loss.T.dot(true_label_dist / n_instances)

    jac_g = measure.jacobian_g(risk)

    # Implementation of multiplication with the loss depends on whether jac_g is an ndarray or spmatrix
    # Result has shape (measure.n_dim_g, n_instances * n_classes)
    if isinstance(jac_g, spmatrix):
        # order below is more efficient if jac_g is a spmatrix
        loss_jac_g_sq = jac_g.dot(loss.transpose())
    else:
        # loss_jac_g_sq = np.einsum('xd,id->ix', loss, jac_g) ** 2
        # order below is more efficient if loss is a spmatrix
        loss_jac_g_sq = loss.dot(jac_g.transpose()).transpose()

    # Square the result, but implementation varies for spmatrix and ndarray
    if isinstance(loss_jac_g_sq, spmatrix):
        loss_jac_g_sq.data[:] = np.square(loss_jac_g_sq.data)
    else:
        loss_jac_g_sq = np.square(loss_jac_g_sq)

    # Sum over components of risk
    if isinstance(loss_jac_g_sq, spmatrix):
        # Result of sum() method is a np.matrix of shape (1, n_instances * n_classes)
        # .A1 extracts the 1d ndarray.
        inner = loss_jac_g_sq.sum(axis=0).A1
    else:
        inner = np.add.reduce(loss_jac_g_sq, axis=0)

    # Take sqrt now for deterministic case
    if deterministic:
        inner = np.sqrt(inner)

    # Apply threshold to ensure correct support
    if epsilon != 0:
        zero_inf_norm = loss.max(axis=1).astype(bool)
        if isinstance(zero_inf_norm, spmatrix):
            zero_inf_norm = zero_inf_norm.toarray()
        zero_inf_norm = zero_inf_norm.squeeze()
        inner = np.maximum(inner, zero_inf_norm * epsilon)

    # Take expectation with respect to label
    pmf = np.reshape(inner * true_label_dist, (n_instances, n_classes))
    pmf = np.add.reduce(pmf, axis=1)

    # Take sqrt now for random case
    if not deterministic:
        pmf = np.sqrt(pmf)

    if normalized:
        pmf = pmf / pmf.sum()
    return pmf


class AdaptiveVarMin(AdaptiveBaseProposal):
    """Adaptive Variance-Minimizing Proposal

    Proposes instances to label by sampling from an adaptive biased proposal
    distribution that seeks to minimize asymptotic variance. The biased
    proposal distribution is approximated using adaptive estimates of the
    oracle response :math:`p(y|x)` for each instance :math:`x` in the pool.

    Parameters
    ----------
    pool : an instance of activeeval.pool.BasePool
        A pool that specifies the support of the proposal.

    measure : an instance of activeeval.measures.BaseMeasure
        The target measure. The proposal is selected so as to minimize
        the asymptotic variance for AIS estimates of this measure.

    oracle_estimator : an instance of activeeval.measure.BaseOE
        An adaptive estimator for the oracle response :math:`p(y|x)`

    epsilon : float, optional (default: 1e-9)
        A positive constant used for thresholding when approximating the
        optimal proposal. Smaller values may result in a closer approximation.

    random_state : int, None or numpy.random.RandomState, optional
        (default = None)
        If int, random_state is the seed used by the random number generator;
        if RandomState instance, random_state is the random number generator;
        if None, the random number generator is the RandomState instance used by
        `numpy.random`.
    """
    def __init__(self, pool: BasePool, measure: BaseMeasure, oracle_estimator: BaseOE,
                 epsilon: float = 1e-9, random_state: Union[int, float, np.random.RandomState, None] = None) -> None:
        super().__init__(pool, random_state=random_state)

        if not isinstance(measure, BaseMeasure):
            raise TypeError("`measure` must be an instance of activeeval.measures.BaseMeasure")
        self.measure = measure

        if not isinstance(oracle_estimator, BaseOE):
            raise TypeError("`oracle_estimator` must be an instance of activeeval.proposals.BaseOE")
        self.oracle_estimator = oracle_estimator

        if not (np.isscalar(epsilon) and epsilon > 0):
            raise ValueError("`epsilon` must be a positive float")
        else:
            self.epsilon = epsilon

        self._pmf = None
        self.update()

    def _draw_impl(self, size: Optional[int]) -> Union[ndarray, int]:
        return self.random_state.choice(self.pool.n_instances, size, p=self._pmf)

    def get_pmf(self, idx: Union[ndarray, int, None] = None) -> Union[ndarray, float]:
        return self._pmf if idx is None else self._pmf[idx]

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

        conditional = self.oracle_estimator.predict()
        self._pmf = compute_optimal_proposal(self.pool, self.oracle_estimator.labels, conditional, self.measure,
                                             epsilon=self.epsilon * self.oracle_estimator.epsilon(),
                                             deterministic=self.oracle_estimator.deterministic)

    def reset(self) -> None:
        self.oracle_estimator.reset()
        self._pmf = None
        self.update()


class StaticVarMin(Passive):
    """Static variance-minimizing proposal

    Proposes instances to label by sampling from a static biased proposal
    distribution that seeks to minimize asymptotic variance. The biased
    proposal distribution is approximated using prior estimates of the
    oracle response :math:`p(y|x)` for each instance :math:`x` in the
    pool.

    Parameters
    ----------
    pool : an instance of activeeval.pool.BasePool
        A pool that specifies the support of the proposal.

    measure : an instance of activeeval.measures.BaseMeasure
        The target measure. The proposal is selected so as to minimize
        the asymptotic variance for AIS estimates of this measure.

    response_est : numpy.ndarray, shape (n_instances, n_classes)
        Estimated oracle response distribution for each instance in the pool.
        The class labels should be indexed in the same order as specified in
        `labels`.

    labels : array-like, shape (n_classes,) or None, (default=None)
        The set of class labels, i.e. the support of the oracle response
        :math:`p(y|x)`. If None, the labels are assumed to be integers in
        the set `{0, 1, ..., n_classes - 1}`, where the number of classes
        `n_classes` is inferred from `response_est`.

    epsilon : float, optional (default=1e-9)
        Probability of sampling from the passive proposal. This should be
        greater than zero to ensure all instances have a non-zero probability
        of being selected.

    deterministic : bool, optional (default=False)
        Whether the oracle is deterministic.

    random_state : int, None or numpy.random.RandomState, optional
        (default = None)
        If int, random_state is the seed used by the random number
        generator; if RandomState instance, random_state is the random
        number generator; if None, the random number generator is the
        RandomState instance used by `numpy.random`.
    """
    def __init__(self, pool: BasePool, measure: BaseMeasure, response_est: Union[Iterable, ndarray],
                 labels: Union[Iterable, ndarray] = None, epsilon: float = 1e-9, deterministic: bool = False,
                 random_state: Union[int, float, np.random.RandomState, None] = None) -> None:
        super().__init__(pool, random_state)

        if not isinstance(measure, BaseMeasure):
            raise TypeError("`measure` must be an instance of activeeval.measures.BaseMeasure")
        self.measure = measure

        self.response_est = response_est
        if response_est.shape[0] != self.pool.n_instances:
            raise ValueError("length of zero-th dimension of `response_est` must be equal to `pool.n_instances`")

        if labels is None:
            self.labels = np.arange(self.response_est.shape[1])
        else:
            self.labels = np.asarray(labels)
        self.deterministic = deterministic

        self.epsilon = float(epsilon)
        if self.epsilon < 0 or self.epsilon > 1:
            raise ValueError("`epsilon` must be on the interval [0, 1]")

        # Compute the optimal pmf using the prior as the true label distribution
        self._optimal_pmf = compute_optimal_proposal(self.pool, self.labels, self.response_est, self.measure,
                                                     epsilon=self.epsilon, deterministic=self.deterministic)

        self._pmf = np.copy(self._optimal_pmf)
        # Mix with uniform
        self._pmf = (1 - self.epsilon) * self._optimal_pmf + self.epsilon / self.pool.n_instances
        # Re-normalize in case of floating point rounding errors
        self._pmf = self._pmf / self._pmf.sum()

    def _draw_impl(self, size: Optional[int]) -> Union[ndarray, int]:
        explore = self.random_state.uniform(size=size) <= self.epsilon
        n_explore = np.sum(explore)
        n_exploit = np.size(explore) - n_explore
        explore_samples = self.random_state.randint(0, self.pool.n_instances, n_explore)
        exploit_samples = self.random_state.choice(self.pool.n_instances, n_exploit, p=self._optimal_pmf)
        if size is None:
            return explore_samples.item() if explore else exploit_samples.item()
        else:
            out = np.empty(size, dtype=int)
            out[explore] = explore_samples
            out[~explore] = exploit_samples
            return out

    def get_pmf(self, idx: Union[ndarray, int, None] = None) -> Union[ndarray, float]:
        return self._pmf if idx is None else self._pmf[idx]

    def get_weight(self, idx: Union[ndarray, int, None] = None) -> Union[ndarray, float]:
        return 1.0 / (self.pool.n_instances * self.get_pmf(idx))
