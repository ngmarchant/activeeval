import numpy as np

from .estimators import AISEstimator, BaseEstimatorWithVariance
from .pools import BasePool
from .measures import BaseMeasure
from .proposals import BaseProposal, AdaptiveBaseProposal
from .estimators import BaseEstimator


from typing import Optional, Union, List, KeysView, Iterable
from numpy import ndarray


class Sample:
    """Container that represents a labeled sample

    Parameters
    ----------
    round_id : int
        The round at which this instance was sampled and labeled

    instance_id : int
        Identifier of an instance

    label : int or str
        Label received for this instance

    weight : float
        Importance weight associated with the sample
    """
    def __init__(self, round_id: int, instance_id: int, label: Union[int, str], weight: float = 1.0):
        self.round_id = round_id
        self.instance_id = instance_id
        self.label = label
        self.weight = weight


class LabelCache:
    """Container for storing labels of instances, as received from the oracle
    """
    def __init__(self, deterministic: bool = True):
        self.deterministic = deterministic
        self._map = {}

    def __getitem__(self, instance: Union[str, int]) -> Union[List[int], int]:
        return self._map[instance]

    def __len__(self) -> int:
        return len(self._map)

    def __str__(self) -> str:
        return str(self._map)

    def __contains__(self, instance_id) -> bool:
        return self._map.__contains__(instance_id)

    def __iter__(self):
        for instance in self._map.items():
            yield instance

    def clear(self) -> None:
        """Clear the cache
        """
        self._map.clear()

    def instances(self) -> KeysView:
        """A view on the instances in the cache
        """
        return self._map.keys()

    def update(self, instance_id, label) -> None:
        """Update cache with a label
        """
        if self.deterministic:
            self._map[instance_id] = label
        else:
            if instance_id in self._map:
                self._map[instance_id].append(label)
            else:
                self._map[instance_id] = [label]


class Evaluator:
    """Runs the adaptive estimation procedure

    Parameters
    ----------
    pool : an instance of activeeval.pool.BasePool
        A pool of unlabeled instances.

    measure : an instance of activeeval.measures.BaseMeasure
        Target measure to estimate.

    proposal : an instance of activeeval.proposals.BaseProposal
        A proposal distribution which is used to selects instances for labeling.
        May be adaptive or static.

    estimator : an instance of activeeval.estimators.BaseMeasureEstimator, optional
        Estimator for the target measure. Defaults to
        :class:`activeeval.estimators.AISEstimator`, an importance-weighted
        estimator that is appropriate for adaptive and static proposals.

    samples : a list of Sample instances, optional
        Sampling history from a previous evaluation on `pool`. These
        samples can be reused to estimate a possibly different
        `measure` on the same pool.

    Attributes
    ----------
    estimate_history : list of numpy.ndarrays
        History of estimates after each label query.

    variance_history : list of numpy.ndarrays or None
        History of variance estimates after each label query. This is only
        available if the estimator is a subclass of
        `activeeval.estimators.BaseEstimatorWithVariance`.

    round_id : int
        Counts the number of rounds.
    """
    def __init__(self, pool: BasePool, measure: BaseMeasure, proposal: BaseProposal,
                 estimator: Optional[BaseEstimator] = None, samples: Optional[List[Sample]] = None,
                 deterministic_oracle: bool = False) -> None:
        if not isinstance(pool, BasePool):
            raise TypeError("`pool` must be an instance of activeeval.pool.BasePool")
        self.pool = pool

        if not isinstance(measure, BaseMeasure):
            raise TypeError("`measure` must be an instance of activeeval.measures.BaseMeasure")
        self.measure = measure

        if estimator is None:
            self.estimator = AISEstimator(measure)
        elif isinstance(estimator, BaseEstimator):
            self.estimator = estimator
        else:
            raise TypeError("`estimator` must be an instance of activeeval.estimators.BaseEstimator or None")

        if not isinstance(proposal, BaseProposal):
            raise TypeError("`proposal` must be an instance of activeeval.proposals.BaseProposal")
        self.proposal = proposal

        self.estimate_history = list()
        if isinstance(self.estimator, BaseEstimatorWithVariance):
            self.variance_history = list()
        else:
            self.variance_history = None

        self.round_id = 0

        if samples is None:
            self.sample_history = list()
        elif isinstance(samples, list):
            # Replay given samples
            self.sample_history = samples
            self.round_id = samples[0].round
            instance_ids = []
            labels = []
            weights = []
            for s in samples:
                if s.round != self.round_id:
                    # Started new round, so update
                    self.update(instance_ids, labels, weights)
                    instance_ids = []
                    labels = []
                    weights = []
                instance_ids.append(s.instance_id)
                labels.append(s.label)
                weights.append(s.weight)

            # Update for last round
            self.update(instance_ids, labels, weights)
        else:
            raise TypeError("`samples` must be a list of `Sample` instances or None")

    def _update_impl(self, instance_id: int, label: Union[int, str], weight: Optional[float]) -> None:
        s = Sample(self.round_id, instance_id, label, weight)
        self.sample_history.append(s)
        self.estimator.update(instance_id, label, weight=weight, proposal=self.proposal)
        self.estimate_history.append(self.estimator.get())
        if isinstance(self.estimator, BaseEstimatorWithVariance):
            var = self.estimator.get_var(self.proposal)
            self.variance_history.append(var)

    def update(self, instance_ids: Union[int, Iterable, ndarray], labels: Union[int, str, Iterable, ndarray],
               weights: Union[float, Iterable, ndarray, None] = None) -> None:
        """Update internal state after a round of labeling instances

        Parameters
        ----------
        instance_ids : int or array-like, shape (n_instances,)
            Identifiers of instances in the pool that were labeled this round.

        labels : int, str or array-like, shape (n_instances,)
            Labels for the instances.

        weights : float or array-like, shape (n_instances,), optional
            Importance weights for the instances, as returned by the
            ``query`` method. If None, the weight for each instance is set to
            1.0.
        """
        if isinstance(self.proposal, AdaptiveBaseProposal):
            self.proposal.update(instance_ids, labels, weights)

        if isinstance(instance_ids, (int, np.integer)):
            self._update_impl(instance_ids, labels, weights)
        elif weights is None:
            for instance_id, label in zip(instance_ids, labels):
                self._update_impl(instance_id, label, None)
        else:
            for instance_id, label, weight in zip(instance_ids, labels, weights):
                self._update_impl(instance_id, label, weight)

        self.round_id += 1

    def query(self, n_samples: Optional[int] = None) -> Union[int, ndarray]:
        """Query instances for a round of labeling

        Parameters
        ----------
        n_samples : int or array-like, optional
            Number of instances to sample for labeling in this round. If
            None, returns a single sample.

        Returns
        -------
        instances_ids : int or numpy.ndarray
            Identifiers of the sampled instances

        weights : float or numpy.ndarray
            Importance weights associated with the sampled instances
        """
        return self.proposal.draw(n_samples)

    @property
    def estimate(self) -> Union[ndarray, float]:
        """Get the current estimate of the target measure
        """
        return self.estimator.get()

    @property
    def var_estimate(self) -> Union[None, ndarray, float]:
        """Get the current estimate of the variance of the target measure, if
        available
        """
        if isinstance(self.estimator, BaseEstimatorWithVariance):
            return self.estimator.get_var(self.proposal)
        else:
            return None

    def reset(self) -> None:
        """Reset the evaluator
        """
        self.estimator.reset()
        self.proposal.reset()
        self.estimate_history = list()
        if isinstance(self.estimator, BaseEstimatorWithVariance):
            self.variance_history = list()
        else:
            self.variance_history = None
        self.sample_history.clear()
        self.round_id = 0
