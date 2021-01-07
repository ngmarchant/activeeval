import numpy as np
import copy
import warnings
from abc import ABC, abstractmethod
from treelib import Tree
from typing import Union, Iterable, Optional
from numpy import ndarray

from ..pools import BasePartitionedPool, BaseHierarchicalPool


class BaseOE(ABC):
    """Base class for an oracle estimator

    Estimates the oracle response :math:`p(y|x)` over instances :math:`x` in
    a pool.

    Parameters
    ----------
    labels : int or array-like, shape (n_classes,)
        Specifies the set of class labels, i.e. the support of the oracle
        response :math:`p(y|x)`. An integer specifies the number of classes
        `n_classes` assuming the labels are in the set
        {0, 1, ..., n_classes - 1}. Otherwise, the complete set of labels
        must be passed as a sequence.

    deterministic : bool, optional (default=True)
        Whether the oracle is deterministic.
    """
    def __init__(self, labels: Union[ndarray, Iterable, int], deterministic: bool = False) -> None:
        if isinstance(labels, (int, np.integer)):
            self.labels = np.arange(labels)
        else:
            self.labels = np.asarray(labels)
        self.deterministic = deterministic

    @abstractmethod
    def _update_impl(self, idx: Optional[ndarray], x: Optional[ndarray], y: Optional[ndarray],
                     weight: Optional[ndarray]) -> None:
        """Class-specific implementation of the update method
        """
        pass

    def update(self, idx: Union[ndarray, int, None, Iterable] = None, x: Union[ndarray, Iterable, None] = None,
               y: Union[ndarray, Iterable, str, int, float, None] = None,
               weight: Union[ndarray, Iterable, float, None] = None) -> None:
        """Update the estimator based on observations

        Parameters
        ----------
        idx : int or array-like with shape (n_instances,) or None, optional (default=None)
            Identifier(s) of the observed instance(s).

        x : array-like with shape (n_features,) or (n_instances, n_features) or None, optional (default=None)
            Feature vector(s) of the observed instance(s).

        y : int, str, float or array-like with shape (n_instances,) or None, optional (default=None)
            Label(s) of the observed instance(s).

        weight: float or array-like with shape (n_instances,) or None, optional (default=None)
            Importance weight(s) associated with the observed instance(s).
        """
        if idx is None and x is None:
            return None
        if idx is not None:
            idx = np.atleast_1d(idx)
        if x is not None:
            x = np.atleast_1d(x)
        if y is not None:
            x = np.atleast_1d(y)
        if weight is not None:
            weight = np.atleast_1d(weight)
        self._update_impl(idx, x, y, weight)

    @abstractmethod
    def _predict_impl(self, idx: Optional[ndarray], x: Optional[ndarray]) -> ndarray:
        """Class-specific implementation of the predict method
        """
        pass

    def predict(self, idx: Union[ndarray, int, Iterable, None] = None,
                x: Union[ndarray, Iterable, None] = None) -> ndarray:
        """Get an estimate of the oracle response distribution at the given
        instances

        Parameters
        ----------
        idx : int or array-like with shape (n_instances,) or None, optional (default=None)
            Identifier(s) of the instance(s) for which p(y|x) is requested.
            If `idx` is specified, then `x` should be None. If both `idx` and
            `x` are unspecified, then predictions are returned for all
            instances.

        x : array-like with shape (n_features,) or (n_instances, n_features) or None, optional (default=None)
            Feature vector(s) of the instance(s) for which p(y|x) is requested.
            If `x` is specified, then `idx` should be None. If both `idx` and
            `x` are unspecified, then predictions are returned for all
            instances.

        Returns
        -------
        conditionals : numpy.ndarray, shape (n_instances, n_classes)
            The conditional distributions corresponding to `idx` or `x`.
        """
        if not (idx is None and x is None):
            if idx is not None and x is not None:
                raise ValueError("`idx` and `x` cannot both be specified")
        if idx is not None:
            idx = np.atleast_1d(idx)
        if x is not None:
            x = np.atleast_1d(x)
        return self._predict_impl(idx, x)

    @abstractmethod
    def epsilon(self) -> float:
        """Get the epsilon parameter, which approaches zero when the
        estimates have converged.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets the estimator (as if no labels have been observed)
        """
        pass


class BasePartitionedOE(BaseOE, ABC):
    """Base class for a partitioned oracle estimator

    Estimates the oracle response :math:`p(y|x)` over instances in a
    partitioned pool.

    Parameters
    ----------
    pool : an instance of activeeval.pool.BasePartitionedPool
        The oracle response is estimated over instances in this partitioned
        pool.

    labels : int or array-like, shape (n_classes,)
        Specifies the set of class labels, i.e. the support of the oracle
        response :math:`p(y|x)`. An integer specifies the number of classes
        `n_classes` assuming the labels are in the set
        {0, 1, ..., n_classes - 1}. Otherwise, the complete set of labels
        must be passed as a sequence.

    deterministic : bool, optional (default=True)
        Whether the oracle is deterministic.

    Attributes
    ----------
    pool : an instance of activeeval.pool.BasePool or None
        The oracle response is estimated over instances in this pool.
    """
    def __init__(self, pool: BasePartitionedPool, labels: Union[ndarray, Iterable, int],
                 deterministic: bool = False) -> None:
        super().__init__(labels, deterministic)
        if not isinstance(pool, BasePartitionedPool):
            raise TypeError("`pool` must be an instance of activeeval.pool.BasePartitionedPool")
        self.pool = pool

    def predict_block(self) -> ndarray:
        pass

    def predict(self, idx: Union[ndarray, int, Iterable, None] = None,
                x: Union[ndarray, Iterable, None] = None) -> ndarray:
        """Get an estimate of the oracle response distribution at the given instances

        Parameters
        ----------
        idx : int or array-like with shape (n_instances,) or None, optional (default=None)
            Identifier(s) of the instance(s) for which p(y|x) is requested.
            If `idx` is unspecified, predictions are returned for all
            instances in the pool.

        x : array-like with shape (n_features,) or (n_instances, n_features) or None, optional (default=None)
            This parameter is ignored for this oracle estimator.

        Returns
        -------
        conditionals : numpy.ndarray, shape (n_instances, n_classes)
            The conditional distributions corresponding to `idx`.
        """
        return super().predict(idx, None)


class PartitionedIndepOE(BasePartitionedOE):
    """Partitioned Independent Oracle Estimator

    Given a partitioned pool into disjoint blocks, this estimator assumes
    the oracle response :math:`p(y|x)` is independent for each block, and
    constant for all instances with a block.

    Parameters
    ----------
    pool : an instance of activeeval.pool.BasePartitionedPool
        The oracle response is estimated over instances in this partitioned
        pool.

    labels : int or array-like, shape (n_classes,)
        Specifies the set of class labels, i.e. the support of the oracle
        response :math:`p(y|x)`. An integer specifies the number of classes
        `n_classes` assuming the labels are in the set
        {0, 1, ..., n_classes - 1}. Otherwise, the complete set of labels
        must be passed as a sequence.

    prior : array-like, shape (n_instances, n_classes) or None, (default=None)
        Prior estimate of the oracle response distribution for each instance
        in the pool. The class labels should be indexed in the same order as
        specified in `labels`. If None, a uniform prior is used for each
        instance.

    prior_strength : float or None, optional (default=None)
        Strength assigned to the prior information in `pool`. Can be
        interpreted as the number of pseudo-observations, spread equally
        across all blocks of the partition. This should be set in
        accordance with the label budget, number of partitions and
        reliability of the prior information. If None, defaults to 2.0
        pseudo-observations per block.

    smoothing_constant : float, optional (default=1.0)
        Positive smoothing constant added to the Dirichlet concentration
        parameters.
    """
    def __init__(self, pool: BasePartitionedPool, labels: Union[ndarray, Iterable, int],
                 prior: Union[ndarray, Iterable, None] = None, prior_strength: Optional[float] = None,
                 smoothing_constant: float = 1.0):
        super().__init__(pool, labels, deterministic=False)

        if prior is None:
            # Use uniform prior
            self.prior = np.full((self.pool.n_instances, self.labels.size), 1.0/self.labels.size, dtype=float)
        else:
            self.prior = prior
            if prior.shape[0] != self.pool.n_instances:
                raise ValueError("length of zero-th dimension of `prior` must be equal to `pool.n_instances`")

        if prior_strength is None:
            self.prior_strength = 2.0 * self.pool.n_blocks
        else:
            self.prior_strength = float(prior_strength)
            if self.prior_strength < 0:
                raise ValueError("`prior_strength` must be non-negative")

        self.smoothing_constant = float(smoothing_constant)
        if self.smoothing_constant <= 0:
            raise ValueError("`smoothing_constant` must be positive")

        # Track the number of rounds (times the update method is called)
        self._n_updates = 0

        # Observed instance counts per block, predicted label and true label. Each instance is only counted once, even
        # if it is sampled multiple times (since labels are assumed deterministic).
        self._obs_counts = np.zeros((self.pool.n_blocks, self.labels.size), dtype=float)

        # Prior counts/weights per block and true label. These are derived from the classifier scores.
        self._prior_counts = (np.array([np.mean(self.prior[block], axis=0) for block in self.pool.blocks_iter()])
                              * self.prior_strength / self.pool.n_blocks)

        # Apply smoothing
        self._prior_counts = self._prior_counts + self.smoothing_constant

    def _update_impl(self, idx: Optional[ndarray], x: Optional[ndarray], y: Optional[ndarray],
                     weight: Optional[ndarray]) -> None:
        block_id = self.pool.block_assignments[idx]
        np.add.at(self._obs_counts, (block_id, y), 1)
        self._n_updates += 1

    def predict_block(self) -> ndarray:
        """Get an estimate of the oracle response within each block

        Returns
        -------
        numpy.ndarray with shape (n_blocks, n_classes)
        """
        # Distribution over labels per block (corresponds to theta in the paper)
        tot_weight = self._obs_counts + self._prior_counts
        return tot_weight / np.add.reduce(tot_weight, axis=1, keepdims=True)

    def _predict_impl(self, idx: Optional[ndarray], x: Optional[ndarray]) -> ndarray:
        block_conditional = self.predict_block()
        block_ids = self.pool.block_assignments if idx is None else self.pool.block_assignments[idx]
        conditional = block_conditional[block_ids, :]
        return conditional

    def epsilon(self) -> float:
        super().epsilon()
        return 1.0 / (self._n_updates + 1.0)

    def reset(self) -> None:
        super().reset()
        self._obs_counts[:] = 0.0
        self._n_updates = 0


class PartitionedStochasticOE(PartitionedIndepOE):
    """Partitioned Stochastic Oracle Estimator

    Given a partitioned pool into disjoint blocks, this estimator assumes
    instances are assigned to blocks conditional on their labels.
    The oracle response :math:`p(y|x)` is then approximated by the response
    at the block-level :math:`p(y|block_idx)` where block_idx is the assigned
    block of the instance. This estimator is suited for stochastic oracles.
    See :func:`activeeval.proposals.PartitionedDeterministicOE` for
    deterministic oracles.

    Parameters
    ----------
    pool : an instance of activeeval.pool.BasePartitionedPool
        The oracle response is estimated over instances in this partitioned
        pool.

    labels : int or array-like, shape (n_classes,)
        Specifies the set of class labels, i.e. the support of the oracle
        response :math:`p(y|x)`. An integer specifies the number of classes
        `n_classes` assuming the labels are in the set
        {0, 1, ..., n_classes - 1}. Otherwise, the complete set of labels
        must be passed as a sequence.

    prior : array-like, shape (n_instances, n_classes) or None, (default=None)
        Prior estimate of the oracle response distribution for each instance
        in the pool. The class labels should be indexed in the same order as
        specified in `labels`. If None, a uniform prior is used for each
        instance.

    prior_strength : float or None, optional (default=None)
        Strength assigned to the prior information in `pool`. Can be
        interpreted as the number of pseudo-observations, spread equally
        across all blocks of the partition. This should be set in
        accordance with the label budget, number of partitions and
        reliability of the prior information. If None, defaults to 2.0
        pseudo-observations per block.

    smoothing_constant : float, optional (default=1.0)
        Positive smoothing constant added to the Dirichlet concentration
        parameters.
    """
    def __init__(self, pool: BasePartitionedPool, labels: Union[ndarray, Iterable, int],
                 prior: Union[ndarray, Iterable, None] = None, prior_strength: Optional[float] = None,
                 smoothing_constant: float = 1.0):
        super().__init__(pool, labels, prior=prior, prior_strength=prior_strength,
                         smoothing_constant=smoothing_constant)

        # Keep track of global counts separately. Set prior from above, before smoothing.
        self._prior_counts_global = (np.add.reduce(self._prior_counts - self.smoothing_constant, axis=0) +
                                     self.smoothing_constant)
        self._obs_counts_global = np.zeros(self.labels.size, dtype=float)

    def _update_impl(self, idx: Optional[ndarray], x: Optional[ndarray], y: Optional[ndarray],
                     weight: Optional[ndarray]) -> None:
        np.add.at(self._obs_counts_global, y, weight)

        block_id = self.pool.block_assignments[idx]
        np.add.at(self._obs_counts, (block_id, y), weight)
        self._n_updates += 1

    def predict_block(self) -> ndarray:
        """Get an estimate of the oracle response within each block

        Returns
        -------
        numpy.ndarray with shape (n_blocks, n_classes)
        """
        # Global label distribution (corresponds to theta in the paper)
        tot_counts_global = self._prior_counts_global + self._obs_counts_global
        global_label_dist = tot_counts_global / np.add.reduce(tot_counts_global, axis=None)

        # Distribution over labels per block (corresponds to theta in the paper)
        tot_weight = self._obs_counts + self._prior_counts
        block_label_dist = tot_weight / np.add.reduce(tot_weight, axis=0, keepdims=True)

        label_dist = np.einsum('y,ky->ky', global_label_dist, block_label_dist)
        return label_dist / np.add.reduce(label_dist, axis=1, keepdims=True)

    def reset(self) -> None:
        super().reset()
        self._obs_counts_global[:] = 0.0


class PartitionedDeterministicOE(PartitionedStochasticOE):
    """Partitioned estimator for a deterministic oracle

    Given a partitioned pool into disjoint blocks, this estimator assumes
    instances are assigned to blocks conditional on their labels.
    The labels of instances are then treated as latent variables, to be
    inferred using the expectation-maximization algorithm. Since each
    instance is assumed to have a single latent label, this estimator is
    only suited for deterministic oracles. See
    :func:`activeeval.proposals.PartitionedStochasticOE` for stochastic
    oracles.

    Parameters
    ----------
    pool : an instance of activeeval.pool.BasePartitionedPool
        The oracle response is estimated over instances in this partitioned
        pool.

    labels : int or array-like, shape (n_classes,)
        Specifies the set of class labels, i.e. the support of the oracle
        response :math:`p(y|x)`. An integer specifies the number of classes
        `n_classes` assuming the labels are in the set
        {0, 1, ..., n_classes - 1}. Otherwise, the complete set of labels
        must be passed as a sequence.

    prior : array-like, shape (n_instances, n_classes) or None, (default=None)
        Prior estimate of the oracle response distribution for each instance
        in the pool. The class labels should be indexed in the same order as
        specified in `labels`. If None, a uniform prior is used for each
        instance.

    prior_strength : float or None, optional (default=None)
        Strength assigned to the prior information in `pool`. Can be
        interpreted as the number of pseudo-observations, spread equally
        across all blocks of the partition. This should be set in
        accordance with the label budget, number of partitions and
        reliability of the prior information. If None, defaults to 2.0
        pseudo-observations per block.

    smoothing_constant : float, optional (default=1.0)
        Positive smoothing constant added to the Dirichlet concentration
        parameters. Must be greater than or equal to 1.0 to ensure MAP
        estimates of the Dirichlet probabilities are defined.

    em_max_iter : int, optional (default=500)
        Maximum number of iterations allowed for the EM algorithm to converge.

    em_tol : float, optional (default=0.0001)
        Tolerance for convergence of the EM algorithm. Once the relative
        change in the log posterior falls below `em_tol` the algorithm is
        terminated.
    """
    def __init__(self, pool: BasePartitionedPool, labels: Union[ndarray, Iterable, int],
                 prior: Union[ndarray, Iterable, None] = None, prior_strength: Optional[float] = None,
                 smoothing_constant: float = 1.0, em_max_iter: int = 500, em_tol: float = 0.0001):
        if smoothing_constant < 1.0:
            raise ValueError("`smoothing_constant` must be greater than or equal to 1.0")

        super().__init__(pool, labels, prior=prior, prior_strength=prior_strength,
                         smoothing_constant=smoothing_constant)
        self.deterministic = True

        if em_max_iter <= 1:
            raise ValueError("`em_max_iter` must be greater than 1")
        self.em_max_iter = em_max_iter

        if em_tol <= 0:
            raise ValueError("`em_tol` must be a positive float")
        self.em_tol = float(em_tol)

        # Record number of unobserved labels per block
        self._n_unobserved = np.copy(self.pool.block_sizes)

        # Record the observed labels
        self._obs_labels = np.full(self.pool.n_instances, -1, dtype=int)

        # Joint distribution of instances per block/label
        self._joint_dist = self._initialize_joint_dist(self._prior_counts, self.pool.block_weights)

        # Un-normalized log posterior
        self._em_log_post = None

    @staticmethod
    def _initialize_joint_dist(prior_counts: ndarray, block_weights: ndarray) -> ndarray:
        """Estimates the joint probability distribution of instances per block and
        label using prior information.

        Parameters
        ----------
        prior_counts : numpy.ndarray, shape (n_blocks, n_classes)
            Prior counts per block and label.

        block_weights : numpy.ndarray, shape (n_blocks,)
            Relative sizes of the blocks.

        Returns
        -------
        joint_dist : numpy.ndarray, shape (n_blocks, n_classes)
        """
        return (prior_counts / np.add.reduce(prior_counts, axis=1, keepdims=True) *
                block_weights[:, np.newaxis])

    def _run_em(self) -> None:
        """Run the EM algorithm until the convergence criterion is satisfied,
        or for the maximum number of iterations.
        """
        rel_diff = np.inf
        iter_ctr = 0
        while rel_diff > self.em_tol:
            if iter_ctr >= self.em_max_iter:
                warnings.warn("reached `em_max_iter` = {} without satisfying convergence criterion: relative "
                              "difference {} is not less than `em_tol` = {}".format(self.em_max_iter, rel_diff,
                                                                                    self.em_tol), RuntimeWarning)
                break
            em_log_post_prev = self._em_log_post
            self._em_log_post = self._run_em_step()
            if em_log_post_prev is None:
                # Can't compute the difference on the first iteration, as there isn't a previous value to compare with
                rel_diff = np.inf
            else:
                # Past first iteration
                rel_diff = (self._em_log_post - em_log_post_prev) / max(em_log_post_prev, self._em_log_post)
                rel_diff = np.abs(rel_diff)
            iter_ctr += 1

    def _run_em_step(self) -> float:
        """Run a single step of the EM algorithm.
        """
        # E-step
        unobs_counts = self._n_unobserved[:, np.newaxis] * self.predict_block()
        unobs_counts_global = np.add.reduce(unobs_counts, axis=0)

        # M-step
        log_posterior = 0.0

        # MAP estimate of global label distribution
        tot_counts_global = self._obs_counts_global + unobs_counts_global + self._prior_counts_global - 1
        global_label_dist = tot_counts_global / np.add.reduce(tot_counts_global, axis=None)

        # Add contribution to log posterior
        log_terms = (tot_counts_global *
                     np.log(global_label_dist, out=np.zeros_like(global_label_dist), where=(tot_counts_global > 0)))
        log_posterior += np.add.reduce(log_terms, axis=None)

        # Distribution over blocks per label
        tot_counts = self._obs_counts + unobs_counts + self._prior_counts - 1
        block_dist = tot_counts / np.add.reduce(tot_counts, axis=0, keepdims=True)
        log_terms = (tot_counts *
                     np.log(block_dist, out=np.zeros_like(block_dist), where=(tot_counts > 0)))
        log_posterior += np.add.reduce(log_terms, axis=None)

        self._joint_dist = global_label_dist[np.newaxis, :] * block_dist
        return log_posterior

    def _update_impl(self, idx: Optional[ndarray], x: Optional[ndarray], y: Optional[ndarray],
                     weight: Optional[ndarray]) -> None:
        new = []
        # Note: must process sequentially
        for i, instance_id in enumerate(idx):
            if self._obs_labels[instance_id] < 0:
                new.append(i)
                self._obs_labels[instance_id] = y[i]

        idx_new = idx[new]
        y_new = y[new]

        np.add.at(self._obs_counts_global, y_new, 1)

        block_id_new = self.pool.block_assignments[idx_new]
        np.add.at(self._obs_counts, (block_id_new, y_new), 1)
        np.add.at(self._n_unobserved, block_id_new, -1)

        self._run_em()

    def predict_block(self) -> ndarray:
        """Get an estimate of the oracle response within each block

        Returns
        -------
        numpy.ndarray with shape (n_blocks, n_classes)
        """
        return self._joint_dist / np.add.reduce(self._joint_dist, axis=1, keepdims=True)

    def _predict_impl(self, idx: Optional[ndarray], x: Optional[ndarray]) -> ndarray:
        if self._em_log_post is None:
            # EM needs to be run for the first time
            self._run_em()
        conditional = super()._predict_impl(idx, x)

        # Fill in any certain labels
        if idx is None:
            obs_idx = np.where(self._obs_labels >= 0)[0]
        else:
            obs_idx = idx[self._obs_labels[idx] >= 0]
        obs_label = self._obs_labels[obs_idx]
        conditional[obs_idx, :] = 0.0
        conditional[obs_idx, obs_label] = 1.0

        return conditional

    def epsilon(self) -> float:
        super().epsilon()
        return self._n_unobserved.sum() / self.pool.n_instances

    def reset(self) -> None:
        super().reset()
        self._obs_labels[:] = -1
        self._n_unobserved = np.copy(self.pool.block_sizes)
        self._em_log_post = None
        self._joint_dist = self._initialize_joint_dist(self._prior_counts, self.pool.block_weights)


def _initialize_internal_tree(n_classes: int, prior_weights: ndarray, pool: BaseHierarchicalPool,
                              tree_prior_type: str, smoothing_constant: float,
                              deterministic: bool) -> Tree:
    """Copies the tree data structure representing the hierarchical
    partition, inserts the Bayesian model parameters, and sets the prior.

    Parameters
    ----------
    pool : an instance of activeeval.pool.BaseHierarchicalPartitionedPool
        A hierarchically-partitioned pool which specifies the support of
        the proposal.

    tree_prior_type: str, {'absolutely_continuous', 'continuous_singular'}
        TODO.

    smoothing_constant : float
        Positive smoothing constant to add to the Dirichlet concentration
        parameters.

    deterministic : bool
        Whether the labels are deterministic. This determines the node
        type.

    Returns
    -------
    tree : treelib.Tree instance
        copy of the tree suitable for doing Bayesian inference
    """
    _tree = copy.deepcopy(pool.tree)
    _leaves = [_tree.get_node(nid) for nid in pool.leaf_node_ids]

    for node in _tree.all_nodes_itr():
        node.data = None

    for k, leaf_nid in enumerate(pool.leaf_node_ids):
        # Get prior label "counts" (important to normalize later after all counts are accumulated)
        prior_counts_labels = prior_weights[k]
        leaf_node = _tree.get_node(leaf_nid)
        leaf_node.data = prior_counts_labels

        # Ascend to the root node incrementing (i) prior label counts for each node; and (ii) number of leaf
        # nodes in scope of node.
        # Store data in a tuple (prior_counts, leaves_counts)
        child_nid = leaf_nid
        node = _tree.parent(child_nid)
        while True:
            if node.data is None:
                node.data = (np.array(prior_counts_labels), 1)
            else:
                node.data = (node.data[0] + prior_counts_labels, node.data[1] + 1)
            child_nid = node.identifier
            if child_nid == _tree.root:
                break
            node = _tree.parent(child_nid)

    for node in _tree.all_nodes_itr():
        child_nids = node.successors(_tree.identifier)
        n_children = len(child_nids)
        child_level = _tree.level(node.identifier) + 1
        if n_children > 0:
            # At this point the node data is a tuple containing two elements (i) an array of un-normalized label
            # counts and (ii) count of leaves under node
            if tree_prior_type == 'absolutely_continuous':
                prior_const = smoothing_constant * child_level ** n_children
            else:
                # Continuous singular
                prior_const = smoothing_constant
            prior_counts = np.full((n_children, n_classes), prior_const, dtype=float)
            leaves_counts = np.zeros(n_children, dtype=int)
            for i, nid in enumerate(child_nids):
                node_data = _tree.get_node(nid).data
                prior_counts[i] += node_data[0]
                leaves_counts[i] = node_data[1]
            if deterministic:
                node.data = DeterministicNode(child_nids, n_classes, prior_counts, leaves_counts)
            else:
                node.data = StochasticNode(child_nids, n_classes, prior_counts, leaves_counts)
        else:
            node.data = None

    return _tree


class HierarchicalDeterministicOE(PartitionedDeterministicOE):
    """Hierarchical partitioned estimator for a deterministic oracle

    TODO

    Parameters
    ----------
    pool : an instance of activeeval.pool.BaseHierarchicalPartitionedPool
        The oracle response is estimated over instances in this
        hierarchically-partitioned pool.

    labels : int or array-like, shape (n_classes,)
        Specifies the set of class labels, i.e. the support of the oracle
        response :math:`p(y|x)`. An integer specifies the number of classes
        `n_classes` assuming the labels are in the set
        {0, 1, ..., n_classes - 1}. Otherwise, the complete set of labels
        must be passed as a sequence.

    prior : array-like, shape (n_instances, n_classes) or None, (default=None)
        Prior estimate of the oracle response distribution for each instance
        in the pool. The class labels should be indexed in the same order as
        specified in `labels`. If None, a uniform prior is used for each
        instance.

    prior_strength : float or None, optional (default=None)
        Strength assigned to the prior information in `pool`. Can be
        interpreted as the number of pseudo-observations, spread equally
        across all blocks of the partition. This should be set in accordance
        with the label budget, number of partitions and reliability of prior
        information. If None, defaults to 2.0 pseudo-observations per block.

    smoothing_constant : float, optional (default=1.0)
        Positive smoothing constant added to the Dirichlet concentration
        parameters. Must be greater than or equal to 1.0 to ensure MAP
        estimates of the Dirichlet probabilities are defined.

    tree_smoothing_constant : float, optional (default=1.0)
        Positive smoothing constant added to the Dirichlet concentration
        parameters associated with the branches of the Dirichlet tree.

    tree_prior_type : str, {'absolutely_continuous', continuous_singular'},
        optional (default='absolutely_continuous')
        TODO.

    em_max_iter : int, optional (default=500)
        Maximum number of iterations allowed for the EM algorithm to converge.

    em_tol : float, optional (default=1e-5)
        Tolerance for convergence of the EM algorithm. Once the relative
        change in the log posterior falls below `em_tol` the algorithm is
        terminated.
    """
    def __init__(self, pool: BaseHierarchicalPool, labels: Union[ndarray, Iterable, int],
                 prior: Union[ndarray, Iterable, None] = None, prior_strength: Optional[float] = None,
                 smoothing_constant: float = 1.0, tree_smoothing_constant: float = 1.0,
                 tree_prior_type: str = 'absolutely_continuous', em_max_iter: int = 500, em_tol: float = 1e-5):
        super().__init__(pool, labels, prior=prior, prior_strength=prior_strength,
                         smoothing_constant=smoothing_constant, em_max_iter=em_max_iter, em_tol=em_tol)
        if not isinstance(pool, BaseHierarchicalPool):
            raise TypeError("`pool` must be an instance of activeeval.pool.BaseHierarchicalPartitionedPool")
        self.pool = pool

        if tree_smoothing_constant <= 0:
            raise ValueError("`tree_smoothing_constant` must be a positive float")
        self.tree_smoothing_constant = float(tree_smoothing_constant)

        self.__tree_prior_type_values = ['absolutely_continuous', 'continuous_singular']
        if tree_prior_type in self.__tree_prior_type_values:
            self.tree_prior_type = tree_prior_type
        else:
            raise ValueError("`tree_prior_type` must be one of {}".format(self.__tree_prior_type_values))

        self._tree = _initialize_internal_tree(self.labels.size, self._prior_counts, self.pool, self.tree_prior_type,
                                               self.tree_smoothing_constant, self.deterministic)

    def _run_em_step(self) -> float:
        # E-step
        unobs_counts = self._n_unobserved[:, np.newaxis] * self.predict_block()
        unobs_counts_global = np.add.reduce(unobs_counts, axis=0)

        # Reset unobserved counts
        for node in self._tree.all_nodes_itr():
            if isinstance(node.data, DeterministicNode):
                node.data.reset_unobs_counts()

        # Propagate unobserved counts up the tree
        for k, nid in enumerate(self.pool.leaf_node_ids):
            # Get parent node identifier
            child_node = self._tree.get_node(nid)
            node = self._tree.parent(nid)
            while True:
                node.data.add_unobs_counts(child_node.identifier, unobs_counts[k])
                child_node = node
                if child_node.identifier == self._tree.root:
                    break
                node = self._tree.parent(node.identifier)

        # M-step
        log_posterior = 0.0

        # MAP estimate of global label distribution
        tot_counts_global = self._obs_counts_global + unobs_counts_global + self._prior_counts_global - 1
        global_label_dist = tot_counts_global / np.add.reduce(tot_counts_global, axis=None)

        # Add contribution to log posterior
        log_terms = (tot_counts_global *
                     np.log(global_label_dist, out=np.zeros_like(global_label_dist), where=(tot_counts_global > 0)))
        log_posterior += np.add.reduce(log_terms)

        # Propagate global label distribution through the Dirichlet trees
        blocks_prob_mass = {}  #: stores the probability mass in the leaf nodes
        in_prob_mass = {self._tree.root: global_label_dist}  #: probability mass entering a level of nodes
        out_prob_mass = {}  #: probability mass leaving a level of nodes
        while in_prob_mass:
            # Loop over nodes in a level of the tree
            for nid, prob_mass in in_prob_mass.items():
                node = self._tree.get_node(nid)
                if node.data is None:
                    # Leaf node
                    blocks_prob_mass[nid] = prob_mass
                else:
                    # Inner node
                    tot_counts_children = node.data.counts - node.data.leaves_counts[:, np.newaxis]
                    label_dist = tot_counts_children / np.add.reduce(tot_counts_children, axis=0, keepdims=True)
                    # Add contribution to log posterior
                    log_terms = (tot_counts_children *
                                 np.log(label_dist, out=np.zeros_like(label_dist), where=(tot_counts_children > 0)))
                    log_posterior += np.add.reduce(log_terms, axis=None)
                    out = {str_id: label_dist[int_id] * prob_mass for str_id, int_id in node.data.str_to_int_id.items()}
                    out_prob_mass = {**out_prob_mass, **out}
            # Descend to next level of tree
            in_prob_mass = out_prob_mass
            out_prob_mass = {}

        self._joint_dist = np.array([blocks_prob_mass[nid] for nid in self.pool.leaf_node_ids])
        return log_posterior

    def predict_block(self) -> ndarray:
        return self._joint_dist / np.add.reduce(self._joint_dist, axis=1, keepdims=True)

    def _update_impl(self, idx: Optional[ndarray], x: Optional[ndarray], y: Optional[ndarray],
                     weight: Optional[ndarray]) -> None:
        new = []
        # Note: must process sequentially
        for i, instance_id in enumerate(idx):
            if self._obs_labels[instance_id] < 0:
                new.append(i)
                self._obs_labels[instance_id] = y[i]

        idx_new = idx[new]
        y_new = y[new]
        block_id_new = np.atleast_1d(self.pool.block_assignments[idx_new])

        np.add.at(self._obs_counts_global, y_new, 1)
        np.add.at(self._obs_counts, (block_id_new, y_new), 1)
        np.add.at(self._n_unobserved, block_id_new, -1)

        # Update tree
        unique_block_id = np.unique(block_id_new)
        for k in unique_block_id:
            # Get parent node identifier
            child_node = self._tree.get_node(self.pool.leaf_node_ids[k])
            node = self._tree.parent(child_node.identifier)
            while True:
                select = np.where(block_id_new == k)[0]
                node.data.update(child_node.identifier, y_new[select])
                child_node = node
                if child_node.identifier == self._tree.root:
                    break
                node = self._tree.parent(node.identifier)

        self._run_em()

    def reset(self) -> None:
        super().reset()
        self._tree = _initialize_internal_tree(self.labels.size, self._prior_counts, self.pool, self.tree_prior_type,
                                               self.tree_smoothing_constant, self.deterministic)


class StochasticNode:
    """Tree node for a stochastic oracle

    Parameters
    ----------
    child_idx : array-like
        Identifiers of the children of this node.

    n_classes: int
        Number of classes.

    prior_counts : ndarray, shape (n_children, n_classes)
        Prior counts per child node and label. The order of the children
        along the first axis must match the order in `child_idx`.

    leaves_counts : ndarray, shape (n_children,)
        Number of leaves each child node.

    Attributes
    ----------
    counts : numpy.ndarray, shape (n_children, n_classes)
        Total counts per child node and label. The order of the children
        along the first axis matches the order in `child_idx`.

    n_children : int
        Number of children connected to this node.

    str_to_int_id : dict
        TODO
    """
    def __init__(self, child_idx: Union[Iterable, ndarray], n_classes: int,
                 prior_counts: ndarray, leaves_counts: ndarray) -> None:
        self.str_to_int_id = {str_id: int_id for int_id, str_id in enumerate(child_idx)}
        self.n_children = len(child_idx)
        self.leaves_counts = leaves_counts

        if n_classes <= 0:
            raise ValueError("`n_classes` must be positive")
        self.n_classes = n_classes

        if not isinstance(prior_counts, ndarray):
            raise TypeError("`prior_counts` must be a numpy.ndarray")
        if prior_counts.shape != (self.n_children, self.n_classes):
            raise ValueError("`prior_counts` must have shape (n_children, n_classes)")
        if np.any(prior_counts < 0):
            raise ValueError("`prior_counts` must be non-negative")
        prior_counts = prior_counts.astype(float)

        # Counts includes the prior and observed
        self._obs_prior_counts = prior_counts

    @property
    def counts(self) -> ndarray:
        return self._obs_prior_counts

    def update(self, child_idx: Union[str, ndarray], y: Union[int, ndarray],
               weight: Union[float, ndarray, None] = None) -> None:
        """
        Update observed counts

        Parameters
        ----------
        child_idx : array-like
            TODO

        y : array-like
            TODO

        weight : array-like
            TODO
        """
        weight = 1.0 if weight is None else weight
        y, child_idx = np.atleast_1d(y, child_idx)
        int_child_idx = [self.str_to_int_id[str_id] for str_id in child_idx]
        np.add.at(self._obs_prior_counts, (int_child_idx, y), weight)


class DeterministicNode(StochasticNode):
    """Tree Node for a deterministic oracle

    Parameters
    ----------
    child_idx : array-like
        Identifiers of the children of this node.

    n_classes: int
        Number of classes.

    prior_counts : ndarray, shape (n_children, n_classes)
        Prior counts per child node and label. The order of the children
        along the first axis must match the order in `child_idx`.

    leaves_counts : ndarray, shape (n_children,)
        Number of leaves each child node.

    Attributes
    ----------
    counts : numpy.ndarray, shape (n_children, n_classes)
        Total counts per child node and label. The order of the children
        along the first axis matches the order in `child_idx`.

    n_children : int
        Number of children connected to this node.

    str_to_int_id : dict
        TODO
    """
    def __init__(self, child_idx: Union[Iterable, ndarray], n_classes: int,
                 prior_counts: ndarray, leaves_counts: ndarray) -> None:
        super().__init__(child_idx, n_classes, prior_counts, leaves_counts)
        self._unobs_counts = np.zeros_like(self._obs_prior_counts, dtype=float)

    @property
    def counts(self) -> ndarray:
        return self._obs_prior_counts + self._unobs_counts

    def add_unobs_counts(self, child_idx: str, unobs_counts: ndarray) -> None:
        """
        Update unobserved counts for a single branch

        Parameters
        ----------
        child_idx : str
            TODO

        unobs_counts : numpy.ndarray, shape (n_classes,)
            Unobserved label counts
        """
        int_child_idx = self.str_to_int_id[child_idx]
        self._unobs_counts[int_child_idx] += unobs_counts

    def reset_unobs_counts(self) -> None:
        """
        Reset the unobserved counts to zero.
        """
        self._unobs_counts[:] = 0.0


class HierarchicalStochasticOE(PartitionedStochasticOE):
    """Hierarchical partitioned estimator for a stochastic oracle

    TODO

    Parameters
    ----------
    pool : an instance of activeeval.pool.BaseHierarchicalPartitionedPool
        The oracle response is estimated over instances in this
        hierarchically-partitioned pool.

    labels : int or array-like, shape (n_classes,)
        Specifies the set of class labels, i.e. the support of the oracle
        response :math:`p(y|x)`. An integer specifies the number of classes
        `n_classes` assuming the labels are in the set
        {0, 1, ..., n_classes - 1}. Otherwise, the complete set of labels
        must be passed as a sequence.

    prior : array-like, shape (n_instances, n_classes) or None, (default=None)
        Prior estimate of the oracle response distribution for each instance
        in the pool. The class labels should be indexed in the same order as
        specified in `labels`. If None, a uniform prior is used for each
        instance.

    prior_strength : float or None, optional (default=None)
        Strength assigned to the prior information in `pool`. Can be
        interpreted as the number of pseudo-observations, spread equally
        across all blocks of the partition. This should be set in
        accordance with the label budget, number of partitions and
        reliability of the prior information. If None, defaults to 2.0
        pseudo-observations per block.

    smoothing_constant : float, optional (default=1.0)
        Positive smoothing constant added to the Dirichlet concentration
        parameters.

    tree_smoothing_constant : float, optional (default=1.0)
        Positive smoothing constant added to the Dirichlet concentration
        parameters associated with the branches of the Dirichlet tree.

    tree_prior_type : str, {'absolutely_continuous', continuous_singular'},
        optional (default='absolutely_continuous')
        TODO.
    """
    def __init__(self, pool: BaseHierarchicalPool, labels: Union[ndarray, Iterable, int],
                 prior: Union[ndarray, Iterable, None] = None, prior_strength: Optional[float] = None,
                 smoothing_constant: float = 1.0, tree_smoothing_constant: float = 1.0,
                 tree_prior_type: str = 'absolutely_continuous'):
        super().__init__(pool, labels, prior=prior, prior_strength=prior_strength,
                         smoothing_constant=smoothing_constant)
        if not isinstance(pool, BaseHierarchicalPool):
            raise TypeError("`pool` must be an instance of activeeval.pool.BaseHierarchicalPartition")
        self.pool = pool

        if tree_smoothing_constant <= 0:
            raise ValueError("`tree_smoothing_constant` must be a positive float")
        self.tree_smoothing_constant = float(tree_smoothing_constant)

        self.__tree_prior_type_values = ['absolutely_continuous', 'continuous_singular']
        if tree_prior_type in self.__tree_prior_type_values:
            self.tree_prior_type = tree_prior_type
        else:
            raise ValueError("`tree_prior_type` must be one of {}".format(self.__tree_prior_type_values))

        # Tree that will contain parameters associated with the Bayesian model.
        self._tree = _initialize_internal_tree(self.labels.size, self._prior_counts, self.pool, self.tree_prior_type,
                                               self.tree_smoothing_constant, self.deterministic)

    def _update_impl(self, idx: Optional[ndarray], x: Optional[ndarray], y: Optional[ndarray],
                     weight: Optional[ndarray]) -> None:
        super()._update_impl(idx, x, y, weight)

        block_id = np.atleast_1d(self.pool.block_assignments[idx])
        # Update tree
        unique_block_id = np.unique(block_id)
        for k in unique_block_id:
            # Get parent node identifier
            child_node = self._tree.get_node(self.pool.leaf_node_ids[k])
            node = self._tree.parent(child_node.identifier)
            while True:
                select = np.where(block_id == k)[0]
                node.data.update(child_node.identifier, y[select], weight[select])
                child_node = node
                if child_node.identifier == self._tree.root:
                    break
                node = self._tree.parent(node.identifier)

    def predict_block(self):
        # Global label distribution (corresponds to theta in the paper)
        tot_counts_global = self._prior_counts_global + self._obs_counts_global
        global_label_dist = tot_counts_global / np.add.reduce(tot_counts_global, axis=None)

        # Propagate global label distribution through the Dirichlet trees
        blocks_prob_mass = {}  #: stores the probability mass in the leaf nodes
        in_prob_mass = {self._tree.root: global_label_dist}  #: probability mass entering a level of nodes
        out_prob_mass = {}  #: probability mass leaving a level of nodes
        while in_prob_mass:
            # Loop over nodes in a level of the tree
            for nid, prob_mass in in_prob_mass.items():
                node = self._tree.get_node(nid)
                if node.data is None:
                    # Leaf node
                    blocks_prob_mass[nid] = prob_mass
                else:
                    # Inner node
                    tot_weight_per_label = node.data.counts
                    label_dist = tot_weight_per_label / np.add.reduce(tot_weight_per_label, axis=0)
                    out = {str_id: label_dist[int_id] * prob_mass for str_id, int_id in node.data.str_to_int_id.items()}
                    out_prob_mass = {**out_prob_mass, **out}
            # Descend to next level of tree
            in_prob_mass = out_prob_mass
            out_prob_mass = {}

        label_dist = np.array([blocks_prob_mass[nid] for nid in self.pool.leaf_node_ids])
        return label_dist / np.add.reduce(label_dist, axis=1, keepdims=True)

    def reset(self) -> None:
        super().reset()
        self._tree = _initialize_internal_tree(self.labels.size, self._prior_counts, self.pool, self.tree_prior_type,
                                               self.tree_smoothing_constant, self.deterministic)
