import numpy as np
from numpy import ndarray
from typing import Union, Iterable
from treelib import Tree, Node

from ._base import BasePartitionedPool, BaseHierarchicalPool
from ._standard import Pool


def even_bin_edges(scores: Union[Iterable, ndarray], n_strata: int, scale: str = 'geometric') -> ndarray:
    """Evenly-spaced bin edges

    Returns bin edges that are evenly spaced on a geometric or linear scale

    Parameters
    ----------
    scores : array, shape = (n_instances,)
        Real-valued scores for each instance.

    n_strata : int
        Number of bins over the range of scores.

    scale : str {'geometric' or 'linear'}, optional
        Whether to use a geometric or linear scale. A geometric scale is
        recommended by Gunnging & Horgan (2004) for scores with skewed
        distributions.

    Returns
    -------
    bin_edges : array of dtype float
        Return the bin edges ``(length(hist)+1)``.

    References
    ----------
    .. [1] Gunning, P. & Horgan, J.M. "A new algorithm for the construction of
       stratum boundaries in skewed populations". In: Survey Methodology 30
       (2004), pp. 159–166.
    """
    scores = np.asarray(scores)

    if scale == 'geometric':
        f = np.geomspace
        # Calculate offset so that minimum score coincides with 1
        delta_s = 0 if scores.min() > 0 else 1 - scores.min()
    elif scale == 'linear':
        f = np.linspace
        # No offset required
        delta_s = 0
    else:
        raise ValueError("`scale` must be 'geometric' or 'linear'")

    # Generate bins
    bins = f(scores.min() + delta_s, scores.max() + delta_s, n_strata + 1)

    bins[0] = -np.inf
    bins[-1] = np.inf

    return bins


def csf_bin_edges(scores: Union[Iterable, ndarray], n_strata: int,
                  bins_hist: Union[int, str, Iterable] = 'auto') -> ndarray:
    """Cumulative square-root frequency (CSF) bin edges

    Parameters
    ----------
    scores : array-like, shape = (n_instances,)
        Real-valued scores for each instance.

    n_strata : int
        Desired number of bins along the range of scores.

    bins_hist : int, str or sequence of scalars, optional
        Specifies the bins to use when estimating the distribution of
        scores using `numpy.histogram`. See documentation for the `bins`
        argument of `numpy.histogram` for further details.

    Returns
    -------
    bin_edges : numpy.ndarray, shape (n_bins + 1,)
        Return the bin edges.

    References
    ----------
    .. [1] Tore Dalenius and Joseph L. Hodges. “Minimum Variance Stratification”.
        In: Journal of the American Statistical Association 54.285 (Mar. 1959),
        pp. 88–101.
    """
    scores = np.asarray(scores)

    if n_strata <= 0 or n_strata >= scores.size:
        raise ValueError("`n_strata` must be a positive integer less than `len(scores)`")

    # approximate distribution of scores -- called F
    counts, score_bins = np.histogram(scores, bins=bins_hist)

    n_bins = counts.size
    if n_strata >= n_bins:
        raise ValueError("Histogram-based estimate of cumulative density is "
                         "too coarse: number of bins ({}) is less than "                         
                         "number of strata ({}). Consider reducing `n_strata`"
                         "or specifying `bins_hist` manually.".format(n_bins, n_strata))

    # generate cumulative dist of sqrt(F)
    sqrt_counts = np.sqrt(counts)
    csf = np.cumsum(sqrt_counts)

    width_csf = csf[-1] / n_strata

    # calculate roughly equal bins on cum sqrt(F) scale
    csf_bins = np.arange(n_strata + 1) * width_csf

    # map cum sqrt(F) bins to score bins
    j = 0
    bins = []
    for (idx, value) in enumerate(csf):
        if j == n_strata or idx == n_bins:
            bins.append(score_bins[-1])
            break
        if value >= csf_bins[j]:
            bins.append(score_bins[idx])
            j += 1

    bins[0] = -np.inf
    bins[-1] = np.inf

    return np.array(bins)


def _get_bin_edges(scores: ndarray, n_strata: int, bins: Union[str, Iterable, ndarray],
                   bins_hist: Union[str, int] = 'auto') -> ndarray:
    """
    Parameters
    ----------
    scores : array, shape = (n_instances,)
        Real-valued scores for each instance.

    n_strata : int
        Desired number of strata, corresponding to bins along the range of
        `scores`.

    bins : str {'geometric', 'linear', 'csf'} or a sequence of scalars
        Specifies a method for determining the edges of the bins. If
        'linear' or 'geometric', the bins are evenly-spaced on a
        linear (or geometric) scale over the range of `scores`. If 'csf',
        the bins are determined using the cumulative square-root frequency
        method of Dalenius and Hodges (1959). The bin edges may be specified
        manually by passing a monotonically-increasing sequence of length
        `n_strata + 1` that includes the rightmost edge of each bin.

    bins_hist : int, str or a sequence of scalars, optional (default = 'auto')
        This parameter is only required if `bins = 'csf'`. It specifies the
        bins for a histogram-based estimate of the cumulative distribution
        of `scores`. This parameter is passed to the `bins` parameter of
        `numpy.histogram`, and is documented at `numpy.histogram_bin_edges`.

    Returns
    -------
    bin_edges : array of dtype float
        Return the bin edges ``(length(hist)+1)``.
    """
    if not isinstance(bins, str):
        # Assume bins is a sequence of scalars
        bin_edges = np.atleast_1d(bins)
        if bin_edges.size != n_strata + 1:
            raise ValueError("`len(bins)={}` is incompatible with "
                             "`n_strata={}`".format(bin_edges.size, n_strata))
        return bin_edges

    if bins == 'csf':
        bin_edges = csf_bin_edges(scores, n_strata, bins_hist)
    elif bins == 'linear':
        bin_edges = even_bin_edges(scores, n_strata, scale='linear')
    elif bins == 'geometric':
        bin_edges = even_bin_edges(scores, n_strata, scale='geometric')
    else:
        raise ValueError("Method '{}' for determining bin edges is invalid".format(bins))
    return bin_edges


class StratifiedPool(Pool, BasePartitionedPool):
    """A stratified pool of unlabeled instances

    Represents a pool of unlabeled instances that are stratified (partitioned)
    according to an auxiliary variable. The strata are defined with respect to
    bins that cover the range of the auxiliary variable. The bin edges may
    be specified manually, or computed using a standard method
    (see `bins` below).

    Parameters
    ----------
    scores : array-like, shape (n_instances,)
        Real-valued scores for instances in the pool.

    n_strata : int
        Target number of strata/blocks.

    features : array-like, shape (n_instances, n_features), optional
        Feature vectors for instances in the pool.

    bins : str {'geometric', 'linear', 'csf'} or a sequence of scalars, optional (default = 'csf')
        Specifies a method for determining the edges of the bins. If
        'linear' or 'geometric', the bins are evenly-spaced on a
        linear (or geometric) scale over the range of `scores`. If 'csf',
        the bins are determined using the cumulative square-root frequency
        method of Dalenius and Hodges (1959). The bin edges may be specified
        manually by passing a monotonically-increasing sequence of length
        `n_strata + 1` that includes the rightmost edge of each bin.

    bins_hist : int, str or a sequence of scalars, optional (default = 'auto')
        This parameter is only required if `bins = 'csf'`. It specifies the
        bins for a histogram-based estimate of the cumulative distribution
        of `scores`. This parameter is passed to the `bins` parameter of
        `numpy.histogram`, and is documented at `numpy.histogram_bin_edges`.

    Attributes
    ----------
    n_instances : int
        Number of instances in the pool.

    n_blocks : int
        Size of the partition, i.e. the number of non-empty blocks.

    block_sizes : numpy.ndarray
        An array of block sizes

    block_weights : numpy.ndarray
        An array of relative block sizes

    block_assignments : numpy.ndarray
        An array of block assignments (integer block ids) for each instance in
        the pool.

    Note
    ----
    In general `n_strata >= n_blocks`, since some of the strata may be empty.
    """
    def __init__(self, scores: Union[Iterable, ndarray], n_strata: int,
                 features: Union[ndarray, Iterable, None] = None, bins: Union[str, Iterable, ndarray] = 'csf',
                 bins_hist: Union[str, int] = 'auto') -> None:
        scores = np.asarray(scores)
        if not scores.ndim == 1:
            raise ValueError("`scores` must be an array of shape (n_instances,)")

        super().__init__(scores.size, features)

        bin_edges = _get_bin_edges(scores, n_strata, bins, bins_hist)

        # Assignments vector with block ids in {1, ..., n_strata}
        self._assignments = np.digitize(scores, bin_edges)
        # Remove empty blocks and start ids at zero {0, ..., n_blocks - 1}
        self._assignments = np.unique(self._assignments, return_inverse=True)[1]

        self._argsort_a = np.argsort(self._assignments)
        d = np.ediff1d(self._assignments[self._argsort_a], to_begin=1, to_end=1)
        self._I = np.repeat(np.arange(d.shape[0]), d)
        self._sizes = np.ediff1d(self._I)

    @property
    def block_assignments(self) -> ndarray:
        return self._assignments

    @property
    def block_sizes(self) -> ndarray:
        return self._sizes

    def block(self, block_id: int) -> ndarray:
        return self._argsort_a[self._I[block_id]:self._I[block_id + 1]]


class HierarchicalStratifiedPool(Pool, BaseHierarchicalPool):
    """A hierarchically-stratified pool of unlabeled instances

    Represents a pool of unlabeled instances that are hierarchically
    stratified (partitioned) according to an auxiliary variable. The strata
    are first defined non-hierarchially with respect to bins that cover
    the range of the auxiliary variable. The bin edges may be specified
    manually, or computed using a standard method (see `bins` below). The
    hierarchical structure is represented using a tree, and is filled
    in using the specified branching structure at each level. The strata
    correspond to the external nodes of the tree in depth-first order.

    Parameters
    ----------
    scores : array-like, shape (n_instances,)
        Real-valued scores for instances in the pool.

    depth : int
        Desired depth of the tree. Must be a positive integer.

    n_children : int or array-like
        Specifies the number of children to insert at each level of the
        tree. If an integer is given, the same number of children is
        inserted at each level. If an array is given, the entries specify
        the number of children at each successive level (must be of
        length `depth`).

    features : array-like, shape (n_instances, n_features), optional
        Feature vectors for instances in the pool.

    bins : str {'geometric', 'linear', 'csf'} or a sequence of scalars, optional (default = 'csf')
        Specifies a method for determining the edges of the bins. If
        'linear' or 'geometric', the bins are evenly-spaced on a
        linear (or geometric) scale over the range of `scores`. If 'csf',
        the bins are determined using the cumulative square-root frequency
        method of Dalenius and Hodges (1959). The bin edges may be specified
        manually by passing a monotonically-increasing sequence of length
        `n_strata + 1` that includes the rightmost edge of each bin.

    bins_hist : int, str or a sequence of scalars, optional (default = 'auto')
        This parameter is only required if `bins = 'csf'`. It specifies the
        bins for a histogram-based estimate of the cumulative distribution
        of `scores`. This parameter is passed to the `bins` parameter of
        `numpy.histogram`, and is documented at `numpy.histogram_bin_edges`.

    Attributes
    ----------
    n_instances : int
        Number of instances in the pool.

    n_blocks : int
        Size of the partition, i.e. the number of non-empty blocks.

    block_sizes : numpy.ndarray
        An array of block sizes

    block_weights : numpy.ndarray
        An array of relative block sizes

    block_assignments : numpy.ndarray
        An array of block assignments (integer block ids) for each instance in
        the pool.

    tree : treelib.Tree instance
        Internal representation of the tree.

    leaf_node_ids : numpy.ndarray of strings
        Leaf node identifiers (corresponding to `tree`) for each block.
    """
    def __init__(self, scores: Union[Iterable, ndarray], depth: int, n_children: Union[Iterable, ndarray, int],
                 features: Union[ndarray, Iterable, None] = None, bins: Union[str, Iterable, ndarray] = 'csf',
                 bins_hist: Union[str, int] = 'auto') -> None:
        scores = np.asarray(scores)
        if not scores.ndim == 1:
            raise ValueError("`scores` must be an array of shape (n_instances,)")

        super().__init__(scores.size, features)

        if not np.isscalar(depth):
            raise TypeError("`depth` must be a scalar")
        elif depth < 0:
            raise ValueError("`depth` must be non-negative")
        else:
            self.depth = depth

        if np.isscalar(n_children):
            self.n_children = np.repeat(n_children, depth)
        else:
            self.n_children = np.asarray(n_children)

        if not self.n_children.shape == (depth,) or not np.can_cast(self.n_children, np.integer):
            raise ValueError("`n_children` must be a scalar integer or an integer array of length `depth`")
        if not np.all(self.n_children > 1):
            raise ValueError("`n_children` must be strictly greater than 1 for all depths")

        n_strata = np.multiply.reduce(self.n_children).item()
        bin_edges = _get_bin_edges(scores, n_strata, bins, bins_hist)

        # Compute tree and populate with instances
        self._tree = self._build_tree(scores, bin_edges=bin_edges)

        # Get leaves in DEPTH order
        self._leaves = []
        self._leaf_nids = []
        for nid in self._tree.expand_tree(mode=self._tree.DEPTH):
            if self._tree[nid].is_leaf():
                self._leaves.append(self._tree.get_node(nid).data)
                self._leaf_nids.append(nid)
        self._leaf_nids = np.asarray(self._leaf_nids)

        # Assignments vector with block ids in {0, ..., n_blocks - 1}
        self._assignments = np.empty(self._n_instances, dtype=int)
        self._sizes = np.empty(len(self._leaves), dtype=int)
        for k, instance_ids in enumerate(self._leaves):
            self._sizes[k] = len(instance_ids)
            self._assignments[instance_ids] = k

    def _build_tree(self, scores: ndarray, bin_edges: ndarray) -> Tree:

        # Build tree with specified number of children at each level
        tree = Tree()
        tree.add_node(Node())  # root node
        nodes_prev = [tree.get_node(tree.root)]
        for level in range(self.depth):
            nodes_current = []
            for node in nodes_prev:
                children = []
                for _ in range(self.n_children[level]):
                    child = Node()
                    tree.add_node(child, parent=node)
                    children.append(child)
                nodes_current.extend(children)
            nodes_prev = nodes_current

        assignments = np.digitize(scores, bin_edges) - 1

        # Store instance ids in leaves
        leaves = tree.leaves()
        for k, node in enumerate(leaves):
            instance_ids = np.where(assignments == k)[0]
            if instance_ids.size == 0:
                tree.remove_node(node.identifier)
            else:
                node.data = instance_ids

        # Prune empty leaves
        check_for_empty_leaves = True
        while check_for_empty_leaves:
            check_for_empty_leaves = False
            leaves = tree.leaves()
            for node in leaves:
                if node.data is None and len(node.successors(tree.identifier)) == 0:
                    # Node is empty and has no siblings
                    tree.remove_node(node.identifier)
                    check_for_empty_leaves = True

        # Simplify tree: remove nodes that only have one child
        for nid in tree.expand_tree(mode=tree.WIDTH):
            children = tree.children(nid)
            if len(children) == 1:
                tree.link_past_node(nid)

        return tree

    @property
    def block_assignments(self) -> ndarray:
        return self._assignments

    @property
    def block_sizes(self) -> ndarray:
        return self._sizes

    @property
    def tree(self) -> Tree:
        return self._tree

    def block(self, block_id: int) -> ndarray:
        return self._leaves[block_id]

    @property
    def leaf_node_ids(self) -> ndarray:
        return self._leaf_nids
