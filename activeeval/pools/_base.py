from abc import ABC, abstractmethod
from typing import Union, Iterator, List, Iterable, Optional
from numpy import ndarray
from treelib import Tree


class BasePool(ABC):
    """Base class for a pool of unlabeled instances

    Attributes
    ----------
    n_instances : int
        Number of instances in the pool.
    """
    def __init__(self) -> None:
        pass

    @property
    @abstractmethod
    def n_instances(self) -> int:
        """Number of instances in the pool
        """
        pass

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i >= self.n_instances:
            raise StopIteration
        out = self[self._i]
        self._i += 1
        return out

    def __len__(self) -> int:
        return self.n_instances

    @abstractmethod
    def __getitem__(self, instance_ids: Union[int, ndarray, Iterable]) -> Optional[ndarray]:
        """Get features for instances in the pool

        Parameter
        ---------
        instance_id : array-like or int
            Integer identifiers of instances to retrieve

        Returns
        -------
        features : numpy.ndarray, shape (n_instances, n_features)
            Feature vectors for the instances, stored in an array. If a
            feature vector is not available, None is returned instead.
        """
        pass


class BasePartitionedPool(BasePool):
    """Base class for a partitioned pool of unlabeled instances

    Represents a pool of unlabeled instances that is partitioned into disjoint
    blocks.

    Attributes
    ----------
    n_instances : int
        Number of instances in the pool.

    n_blocks : int
        Number of non-empty blocks.

    block_sizes : numpy.ndarray
        An array of block sizes.

    block_weights : numpy.ndarray
        An array of relative block sizes.

    block_assignments : numpy.ndarray
        An array of block assignments (integer block ids) for each instance in
        the pool.
    """
    def __init__(self) -> None:
        super().__init__()

    @property
    def n_blocks(self) -> int:
        """Returns the number of (non-empty) blocks
        """
        return self.block_sizes.size

    @abstractmethod
    def block(self, block_id: int) -> Union[List[ndarray], ndarray]:
        """Get the instances assigned to a block

        Parameters
        ----------
        block_id : int
            Integer identifier of the block

        Returns
        -------
        block : numpy.ndarray
            An array containing the identifiers of instances assigned to the
            block
        """
        pass

    @property
    @abstractmethod
    def block_sizes(self) -> ndarray:
        """Get the block sizes of the partition as a numpy.ndarray with
        shape (n_blocks,)
        """
        pass

    @property
    def block_weights(self) -> ndarray:
        """Get the relative block sizes of the partition as a numpy.ndarray
        with shape (n_blocks,)
        """
        return self.block_sizes / self.n_instances

    @property
    @abstractmethod
    def block_assignments(self) -> ndarray:
        """Get the block assignments for the instances as a numpy.ndarray
        with shape (n_instances,)
        """
        pass

    def blocks_iter(self) -> Iterator[ndarray]:
        """An iterator over the blocks"""
        for block_id in range(self.n_blocks):
            yield self.block(block_id)


class BaseHierarchicalPool(BasePartitionedPool):
    """Base class for a hierarchical pool of unlabeled instances

    Represents a pool of unlabeled instances that is hierarchically
    partitioned into disjoint blocks

    Attributes
    ----------
    n_instances : int
        Number of instances in the pool.

    n_blocks : int
        Number of non-empty blocks at the bottom of the hierarchy.

    block_sizes : numpy.ndarray
        An array of block sizes.

    block_weights : numpy.ndarray
        An array of relative block sizes.

    block_assignments : numpy.ndarray
        An array of block assignments (integer block ids) for each instance in
        the pool.

    tree : treelib.Tree instance
        Internal representation of the tree.

    leaf_node_ids : numpy.ndarray of strings
        Leaf node identifiers (corresponding to `tree`) for each block.
    """
    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def tree(self) -> Tree:
        pass

    @property
    @abstractmethod
    def leaf_node_ids(self) -> ndarray:
        pass
