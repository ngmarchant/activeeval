from abc import ABC
import warnings
import numpy as np
from typing import Union, Optional, Iterable
from numpy import ndarray

from ..proposals import BaseProposal
from ..measures import BaseMeasure


class BaseEstimator(ABC):
    """Base Class for an Estimator

    Parameters
    ----------
    measure : an instance of activeeval.measures.BaseMeasure
        A target generalized measure to estimate.
    """
    def __init__(self, measure: BaseMeasure) -> None:
        if not isinstance(measure, BaseMeasure):
            raise TypeError("`measure` must be an instance of activeeval.measures.BaseMeasure")
        self.measure = measure
        self.n_samples = 0

    def update(self, idx: Union[int, ndarray, Iterable], y: Union[int, float, ndarray, Iterable],
               weight: Union[float, ndarray, Iterable] = 1.0, x: Union[int, float, ndarray, None] = None,
               proposal: Optional[BaseProposal] = None) -> None:
        """Update the estimator

        Parameters
        ----------
        idx : integer or array-like, shape (n_instances,)
            Identifiers of observed instances.

        y : integer or array-like, shape (n_instances,)
            Labels of observed instances.

        weight : float or array-like, shape = (n_instances,)
            Importance weights associated with the observed instances.

        x : array-like, shape (n_instances, n_features), or None, optional
            (default=None)
            Feature vectors associated with the observed instances.

        proposal : an instance of activeeval.proposals.BaseProposal or None, optional (default=None)
            Proposal used to obtain the observations.
        """
        if not (isinstance(idx, (np.integer, int)) or (idx is None)):
            if np.shape(idx)[0] != np.shape(y)[0]:
                raise ValueError("mismatch between shape of `idx` and `y`")
            if np.shape(idx)[0] != np.shape(weight)[0]:
                raise ValueError("mismatch between shape of `idx` and `weight`")

    def get(self) -> Union[ndarray, float]:
        """Get the current estimate of the target measure.

        Returns
        -------
        estimate : float or numpy.ndarray, shape (n_dim_risk,)
            Estimate of the target measure. If n_dim_risk is 1, returns a
            float, otherwise returns an array.
        """
        if self.n_samples == 0:
            warnings.warn("cannot produce an estimate without any samples", UserWarning)
            return np.full(self.measure.n_dim_g, fill_value=np.nan)

    def reset(self) -> None:
        """Reset the estimator
        """
        self.n_samples = 0


class BaseEstimatorWithVariance(BaseEstimator):
    """Base class for a generalized measure estimator with support for
    producing variance estimates

    Parameters
    ----------
    measure : an instance of activeeval.measures.BaseMeasure
        The target generalized measure.
    """
    def __init__(self, measure: BaseMeasure) -> None:
        super().__init__(measure)

    def get_var(self, proposal: Optional[BaseProposal] = None) -> Union[None, float, ndarray]:
        """Get the current estimate of the estimator variance

        Parameters
        ----------
        proposal : an instance of activeeval.proposals.BaseProposal or None, optional (default=None)
            Latest estimate of the target proposal. May be required to
            estimate the variance for some estimators.

        Returns
        -------
        variance : numpy.ndarray, shape (n_dim_risk,)
            Estimate of estimator variance.
        """
        if proposal is not None and not isinstance(proposal, BaseProposal):
            raise TypeError("`proposal` must be an instance of activeeval.proposals.BaseProposal or None")
        if self.n_samples == 0:
            warnings.warn("cannot produce an estimate without any samples", UserWarning)
            return np.full((self.measure.n_dim_g, self.measure.n_dim_g), fill_value=np.nan)
