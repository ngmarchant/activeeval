import warnings
import numpy as np
from typing import Union, Optional, Iterable
from numpy import ndarray
from scipy.sparse import spmatrix

from ..proposals import BaseProposal
from ..measures import BaseMeasure
from ._base import BaseEstimator, BaseEstimatorWithVariance


class AISEstimator(BaseEstimator):
    """Adaptive Importance Sampling (AIS) Estimator

    This estimator is suitable for adaptive, biased proposals. It corrects
    for the bias using simple importance re-weighting. After :math:`J`
    samples, the estimate for the target measure :math:`G` is given by:

    .. math::

        G^{\\mathrm{AIS}} = g(\\hat{R}^{\\mathrm{AIS}})

        \\mathrm{where} \\quad \\hat{R}^{\\mathrm{AIS}} = \\frac{1}{J} \\sum_{j = 1}^{J} w_j \\ell(x_j, y_j; f)

    and :math:`w_j` are the importance weights.

    Parameters
    ----------
    measure : an instance of activeeval.measures.BaseMeasure
        A target generalized measure to estimate.
    """
    def __init__(self, measure: BaseMeasure) -> None:
        super().__init__(measure)
        self._risk_numerator = np.zeros(measure.n_dim_risk, dtype=float)

    def update(self, idx: Union[int, ndarray, Iterable], y: Union[int, float, ndarray, Iterable],
               weight: Union[float, ndarray, Iterable] = 1.0, x: Union[int, float, ndarray, None] = None,
               proposal: Optional[BaseProposal] = None) -> None:
        super().update(idx, y, weight, x, proposal)
        idx, y, weight = np.atleast_1d(idx, y, weight)
        x = np.atleast_1d(x) if x is not None else x
        loss = self.measure.loss(idx, y, x)
        if isinstance(loss, spmatrix):
            self._risk_numerator += loss.multiply(weight[:, np.newaxis]).sum(axis=0).A1
        else:
            self._risk_numerator += np.add.reduce(loss * weight[:, np.newaxis], axis=0)
        self.n_samples += y.shape[0]

    def get(self) -> Union[ndarray, float]:
        super().get()
        measure = self.measure.g(self._risk_numerator / self.n_samples)
        try:
            return measure.item()
        except ValueError:
            return measure

    def reset(self) -> None:
        super().reset()
        self._risk_numerator = np.zeros(self.measure.n_dim_risk, dtype=float)


class DeterministicWeightedAISEstimator(BaseEstimator):
    """Adaptive Importance Sampling (AIS) Estimator with Deterministic Weighting

    This estimator is based on work by Owen & Zhou (2020), and is suitable for
    multi-stage AIS schemes. It combines estimates from each stage, placing
    more weight on later stages (where the variance has hopefully improved).
    At stage :math:`t` an estimate of the risk is obtained as follows:

    .. math::

        \\hat{R}_t = \\frac{1}{N_t} \\sum_{i = 1}^{N_t} w_i \\ell(x_i, y_i)

    where :math:`N_t` denotes the number of samples drawn in stage
    :math:`t`. The risk estimates from all :math:`T` stages are then
    combined as follows:

    .. math::

        \\hat{R} = \\frac{\\sum_{t = 1}^{T} \\sqrt{t} \\hat{R}_t}{\\sum_{t = 1}^{T} \\sqrt{t}}.

    This is then plugged into the mapping function :math:`g` to produce an
    estimate of the target measure :math:`\\hat{G} = g(\\hat{R})`.

    Parameters
    ----------
    measure : an instance of activeeval.measures.BaseMeasure
        A target generalized measure to estimate.

    References
    ----------
    .. [1] Owen, A. B. and Zhou, Y. (2020). "The Square Root Rule for Adaptive
      Importance Sampling." *ACM Trans. Model. Comput. Simul.* **30** (2),
      Article 13, DOI: `10.1145/3350426 <https://doi.org/10.1145/3350426>`_.
    """
    def __init__(self, measure: BaseMeasure) -> None:
        super().__init__(measure)
        self._numerator = np.zeros(measure.n_dim_risk, dtype=float)
        self._denominator = 0.0
        self._stage = 0

    def update(self, idx: Union[int, ndarray, Iterable], y: Union[int, float, ndarray, Iterable],
               weight: Union[float, ndarray, Iterable] = 1.0, x: Union[int, float, ndarray, None] = None,
               proposal: Optional[BaseProposal] = None) -> None:
        super().update(idx, y, weight, x, proposal)
        idx, y, weight = np.atleast_1d(idx, y, weight)
        x = np.atleast_1d(x) if x is not None else x
        loss = self.measure.loss(idx, y, x)
        if isinstance(loss, spmatrix):
            risk_numerator = loss.multiply(weight[:, np.newaxis]).sum(axis=0).A1
        else:
            risk_numerator = np.add.reduce(loss * weight[:, np.newaxis], axis=0)
        self._stage += 1
        self._numerator += np.sqrt(self._stage) * (risk_numerator / y.shape[0])
        self._denominator += np.sqrt(self._stage)
        self.n_samples += y.shape[0]

    def get(self) -> Union[ndarray, float]:
        super().get()
        measure = self.measure.g(self._numerator / self._denominator)
        try:
            return measure.item()
        except ValueError:
            return measure

    def reset(self) -> None:
        super().reset()
        self._numerator = np.zeros(self.measure.n_dim_risk, dtype=float)
        self._denominator = 0.0
        self._stage = 0


class AISEstimatorWithVariance(BaseEstimatorWithVariance):
    """AIS Estimator with Support for Variance Estimates

    This is the same estimator as :class:`activeeval.estimators.AISEstimator`
    with the added capability of providing asymptotic variance estimates.
    This comes at the cost of additional memory and computational overhead.

    Parameters
    ----------
    measure : an instance of activeeval.measures.BaseMeasure
        The target measure.
    """
    def __init__(self, measure: BaseMeasure) -> None:
        super().__init__(measure)
        self._risk_numerator = np.zeros(measure.n_dim_risk, dtype=float)
        self._hist_w = []
        self._hist_loss = []
        self._hist_idx = []
        self._var = 0.0

    def update(self, idx: Union[int, ndarray, Iterable], y: Union[int, float, ndarray, Iterable],
               weight: Union[float, ndarray, Iterable] = 1.0, x: Union[int, float, ndarray, None] = None,
               proposal: Optional[BaseProposal] = None) -> None:
        # Copied from AISEstimator.update
        super().update(idx, y, weight, x, proposal)
        idx, y, weight = np.atleast_1d(idx, y, weight)
        x = np.atleast_1d(x) if x is not None else x
        loss = self.measure.loss(idx, y, x)
        if isinstance(loss, spmatrix):
            self._risk_numerator += loss.multiply(weight[:, np.newaxis]).sum(axis=0).A1
        else:
            self._risk_numerator += np.add.reduce(loss * weight[:, np.newaxis], axis=0)
        self.n_samples += y.shape[0]

        # Update variance quantities
        self._hist_idx.extend(idx.tolist())
        self._hist_w.extend(weight.tolist())
        self._hist_loss.extend(loss.tolist())

    def get(self) -> Union[ndarray, float]:
        super().get()
        measure = self.measure.g(self._risk_numerator / self.n_samples)
        try:
            return measure.item()
        except ValueError:
            return measure

    def get_var(self, proposal: Optional[BaseProposal] = None) -> Union[None, float, ndarray]:
        super().get_var(proposal=proposal)
        if proposal is None:
            raise ValueError("`proposal` is required to estimate the variance")
        hist_idx = np.array(self._hist_idx)
        hist_w = np.array(self._hist_w)
        hist_loss = np.array(self._hist_loss)
        hist_w_inf = proposal.get_weight(hist_idx)
        hist_w_inf[~np.isfinite(hist_w_inf)] = 0  # exclude terms that have zero weight wrt q_inf
        risk = self._risk_numerator / self.n_samples
        jac = self.measure.jacobian_g(risk)
        # TODO check dimensions and type
        first_term = np.mean((hist_loss @ jac.T)**2 * hist_w * hist_w_inf)
        second_term = - (risk @ jac.T) ** 2
        return (first_term + second_term) / self.n_samples

    def reset(self) -> None:
        super().reset()
        self._risk_numerator = np.zeros(self.measure.n_dim_risk, dtype=float)
        self._hist_w = []
        self._hist_loss = []
        self._hist_idx = []
        self._var = 0.0
