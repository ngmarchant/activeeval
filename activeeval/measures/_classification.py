import warnings
import numpy as np
from typing import Union, Optional, Iterable
from numpy import ndarray
from scipy.sparse import bmat, spdiags, spmatrix, csr_matrix, hstack

from ._base import BaseMeasure
from ..input_validation import coerce_to_1d, assert_finite


class FMeasure(BaseMeasure):
    """F-measure

    F-measure is the weighted harmonic mean of precision and recall. It is
    defined as:

    .. math::

        F_{\\beta} = \\frac{(1 + \\beta^2) P R}{\\beta^2 P + R}

    where :math:`P` is the precision, :math:`R` is the recall and
    :math:`\\beta` is a positive weight.

    Parameters
    ----------
    y_pred : array-like, shape=(n_instances,)
        Predicted labels from the classifier under evaluation.

    beta : float, optional (default=1.0)
        Weight associated with recall. The default value of 1.0 corresponds
        to the balanced F-measure (a.k.a. F1 score).

    pos_label : int or str, optional (default=1)
        Label used to represent the positive class.

    Notes
    -----
    F-measure can be expressed as a generalized measure by setting the loss
    function as follows:

    .. math::

        \\ell(x, y; f) = \\left[
            \\mathbb{I}[f(x) \\neq y] ,
            (1 - \\alpha) \\mathbb{I}[y = y_+] + \\alpha \\mathbb{I}[f(x) = y_+]
        \\right]

    where :math:`f(x)` is the predicted label for instance :math:`x`,
    :math:`y_+` is the label of the positive class,
    :math:`\\alpha = 1/(1 + \\beta^2)`, and :math:`\\mathbb{I}[\\cdot]` is
    the indicator function.
    With this definition of the loss function, the mapping function must be
    set as:

    .. math::

        g(R) = \\frac{R_1}{R_2}

    where :math:`R = \\mathbb{E}[\\ell(X, Y; f)]` denotes the risk.
    """

    def __init__(self, y_pred: Union[ndarray, Iterable], beta: float = 1.0, pos_label: Union[int, str] = 1) -> None:
        super().__init__()
        self.y_pred = coerce_to_1d(y_pred)
        self.beta = beta
        self._alpha = 1 / (1 + beta ** 2)

        self.pos_label = pos_label

        self.n_dim_risk = 2
        self.n_dim_g = 1

    def loss(self, idx: Union[int, ndarray, Iterable], y: Union[int, float, str, ndarray, Iterable],
             x: Union[int, float, ndarray, Iterable, None] = None) -> Union[ndarray, spmatrix]:
        positive = (y == self.pos_label) * 1
        pred_positive = (self.y_pred[idx] == self.pos_label) * 1
        true_positive = positive * pred_positive
        return np.c_[true_positive, (1 - self._alpha) * positive + self._alpha * pred_positive]

    def g(self, risk: ndarray) -> ndarray:
        self._check_risk(risk)
        # Switch off warnings since denominator may be zero
        with np.errstate(divide='ignore', invalid='ignore'):
            out = risk[0] / risk[1]
            return np.atleast_1d(out)

    def jacobian_g(self, risk: ndarray) -> Union[ndarray, spmatrix]:
        self._check_risk(risk)
        # Switch off warnings since denominator may be zero
        with np.errstate(divide='ignore', invalid='ignore'):
            out = np.c_[1 / risk[1], - risk[0] / (risk[1] ** 2)]
            return out


class Precision(FMeasure):
    """Precision

    Precision is the fraction of true positives among predicted positive
    instances.

    Parameters
    ----------
    y_pred : array-like, shape=(n_instances,)
        Predicted labels from the classifier under evaluation.

    pos_label : int, optional (default=1)
        Label used to represent the positive class.

    Notes
    -----
    Precision can be expressed as a generalized measure by setting the loss
    function as follows:

    .. math::

        \\ell(x, y; f) = \\left[
            \\mathbb{I}[f(x) \\neq y] ,
            \\mathbb{I}[f(x) = y_+]
        \\right]

    where :math:`f(x)` is the predicted label for instance :math:`x`,
    :math:`y_+` is the label of the positive class, and
    :math:`\\mathbb{I}[\\cdot]` is the indicator function.
    With this definition of the loss function, the mapping function must be
    set as:

    .. math::

        g(R) = \\frac{R_1}{R_2}

    where :math:`R = \\mathbb{E}[\\ell(X, Y; f)]` denotes the risk.
    """
    def __init__(self, y_pred: Union[ndarray, Iterable], pos_label: Union[int, str] = 1) -> None:
        super().__init__(y_pred, beta=1.0, pos_label=pos_label)

    def loss(self, idx: Union[int, ndarray, Iterable], y: Union[int, float, str, ndarray, Iterable],
             x: Union[int, float, ndarray, Iterable, None] = None) -> Union[ndarray, spmatrix]:
        positive = (y == self.pos_label) * 1
        pred_positive = (self.y_pred[idx] == self.pos_label) * 1
        true_positive = positive * pred_positive
        return np.c_[true_positive, pred_positive]


class Recall(FMeasure):
    """Recall

    Recall is the fraction of true positives among positive instances.

    Parameters
    ----------
    y_pred : array-like, shape=(n_instances,)
        Predicted labels from the classifier under evaluation.

    pos_label : int, optional (default=1)
        Label used to represent the positive class.

    Notes
    -----
    Recall can be expressed as a generalized measure by setting the loss
    function as follows:

    .. math::

        \\ell(x, y; f) = \\left[
            \\mathbb{I}[f(x) \\neq y] ,
            \\mathbb{I}[y = y_+]
        \\right]

    where :math:`f(x)` is the predicted label for instance :math:`x`,
    :math:`y_+` is the label of the positive class, and
    :math:`\\mathbb{I}[\\cdot]` is the indicator function.
    With this definition of the loss function, the mapping function must be
    set as:

    .. math::

        g(R) = \\frac{R_1}{R_2}

    where :math:`R = \\mathbb{E}[\\ell(X, Y; f)]` denotes the risk.
    """

    def __init__(self, y_pred: Union[ndarray, Iterable], pos_label: Union[int, str] = 1) -> None:
        super().__init__(y_pred, beta=0.0, pos_label=pos_label)

    def loss(self, idx: Union[int, ndarray, Iterable], y: Union[int, float, str, ndarray, Iterable],
             x: Union[int, float, ndarray, Iterable, None] = None) -> Union[ndarray, spmatrix]:
        positive = (y == self.pos_label) * 1
        pred_positive = (self.y_pred[idx] == self.pos_label) * 1
        true_positive = positive * pred_positive
        return np.c_[true_positive, positive]


class Accuracy(BaseMeasure):
    """Accuracy

    Accuracy is the fraction of instances for which the predicted label
    matches the true label.

    Parameters
    ----------
    y_pred : array-like, shape=(n_instances,)
        Predicted labels from the classifier under evaluation.

    Notes
    -----
    Accuracy can be expressed as a generalized measure by setting the loss
    function as follows:

    .. math::

        \\ell(x, y; f) = \\left[ \\mathbb{I}[f(x) \\neq y] \\right]

    where :math:`f(x)` is the predicted label for instance :math:`x`, and
    :math:`\\mathbb{I}[\\cdot]` is the indicator function.
    With this definition of the loss function, the mapping function must be
    set as:

    .. math:: g(R) = 1 - R

    where :math:`R = \\mathbb{E}[\\ell(X, Y; f)]` denotes the risk.
    """

    def __init__(self, y_pred: Union[ndarray, Iterable]) -> None:
        super().__init__()
        self.y_pred = coerce_to_1d(y_pred)
        self.n_dim_risk = 1
        self.n_dim_g = 1

    def loss(self, idx: Union[int, ndarray, Iterable], y: Union[int, float, str, ndarray, Iterable],
             x: Union[int, float, ndarray, Iterable, None] = None) -> Union[ndarray, spmatrix]:
        return np.c_[(y != self.y_pred[idx]) * 1]

    def g(self, risk: ndarray) -> ndarray:
        self._check_risk(risk)
        return np.atleast_1d(1.0 - risk)

    def jacobian_g(self, risk: ndarray) -> Union[ndarray, spmatrix]:
        self._check_risk(risk)
        return -np.ones((self.n_dim_risk, self.n_dim_g), dtype=float)


class BalancedAccuracy(BaseMeasure):
    """Balanced Accuracy

    Balanced accuracy is the average of specificity (true negative rate) and
    sensitivity (true positive rate).

    Parameters
    ----------
    y_pred : array-like, shape=(n_instances,)
        Predicted labels from the classifier under evaluation.

    pos_label : int or str, optional (default=1)
        Label used to represent the positive class.

    Notes
    -----
    Balanced accuracy can represented as a generalized measure by defining the
    loss function as follows:

    .. math::

        \\ell(x, y; f) = \\left[ \\mathbb{I}[f(x) \\neq y],
                                 \\mathbb{I}[y = y_+],
                                 \\mathbb{I}[f(x) = y_+] \\right]

    where :math:`f(x)` is the predicted label for instance :math:`x`,
    :math:`y_+` is the label of the positive class, and
    :math:`\\mathbb{I}[\\cdot]` is the indicator function.
    With this definition of the loss function, the mapping function must be
    set as:

    .. math::

        g(R) = \\frac{R_1 + R_2 (1 - R_2 - R_3)}{2 R_2 ( 1 - R_2)}

    where :math:`R = \\mathbb{E}[\\ell(X, Y; f)]` denotes the risk.
    """

    def __init__(self, y_pred: Union[ndarray, Iterable], pos_label: Union[int, str] = 1) -> None:
        super().__init__()
        self.y_pred = coerce_to_1d(y_pred)
        self.pos_label = pos_label
        self.n_dim_risk = 3
        self.n_dim_g = 1

    def loss(self, idx: Union[int, ndarray, Iterable], y: Union[int, float, str, ndarray, Iterable],
             x: Union[int, float, ndarray, Iterable, None] = None) -> Union[ndarray, spmatrix]:
        positive = (y == self.pos_label) * 1
        pred_positive = (self.y_pred[idx] == self.pos_label) * 1
        true_positive = positive * pred_positive
        return np.c_[true_positive, positive, pred_positive]

    def g(self, risk: ndarray) -> ndarray:
        self._check_risk(risk)
        # Switch off warnings since denominator may be zero
        with np.errstate(divide='ignore', invalid='ignore'):
            out = (risk[0] + risk[1] * (1 - risk[1] - risk[2])) / (2 * risk[1] * (1 - risk[1]))
            return np.atleast_1d(out)

    def jacobian_g(self, risk: ndarray) -> Union[ndarray, spmatrix]:
        self._check_risk(risk)
        # Switch off warnings since denominator may be zero
        with np.errstate(divide='ignore', invalid='ignore'):
            out = np.c_[0.5 / (risk[1] * (1 - risk[1])),
                        0.5 * (risk[0] * (2 * risk[1] - 1) - risk[1]**2 * risk[2]) / ((1 - risk[1])**2 * risk[1]**2),
                        0.5 / (risk[1] - 1)]
            return out


class MatthewsCorrCoef(BaseMeasure):
    """Matthews Correlation Coefficient (MCC)

    Matthews Correlation Coefficient (MCC) measures the correlation
    between the true and predicted labels for binary classification. It
    returns values on the interval :math:`[-1, 1]`.

    Parameters
    ----------
    y_pred : array-like, shape=(n_instances,)
        Predicted labels from the classifier under evaluation.

    pos_label : int or str, optional (default=1)
        Label used to represent the positive class.

    Notes
    -----
    MCC can be expressed as a generalized measure by defining the loss
    function as follows:

    .. math::

        \\ell(x, y) = \\left[ \\mathbb{I}[f(x) \\neq y],
                              \\mathbb{I}[y = y_+],
                              \\mathbb{I}[f(x) = y_+] \\right]

    where :math:`f(x)` is the predicted label for instance :math:`x`,
    :math:`y_+` is the label of the positive class, and
    :math:`\\mathbb{I}[\\cdot]` is the indicator function.
    With this definition of the loss function, the mapping function must be
    set as:

    .. math::

        g(R) = \\frac{R_1 - R_2 R_3}{\\sqrt{R_2 R_3 (1 - R_2) (1 - R_3)}}

    where :math:`R = \\mathbb{E}[\\ell(X, Y; f)]` denotes the risk.
    """

    def __init__(self, y_pred: Union[ndarray, Iterable], pos_label: Union[int, str] = 1) -> None:
        super().__init__()
        self.y_pred = coerce_to_1d(y_pred)
        self.pos_label = pos_label
        self.n_dim_risk = 3
        self.n_dim_g = 1

    def loss(self, idx: Union[int, ndarray, Iterable], y: Union[int, float, str, ndarray, Iterable],
             x: Union[int, float, ndarray, Iterable, None] = None) -> Union[ndarray, spmatrix]:
        positive = (y == self.pos_label) * 1
        pred_positive = (self.y_pred[idx] == self.pos_label) * 1
        true_positive = positive * pred_positive
        return np.c_[true_positive, positive, pred_positive]

    def g(self, risk: ndarray) -> ndarray:
        self._check_risk(risk)
        risk_1_risk_2 = risk[1] * risk[2]
        # Switch off warnings since denominator may be zero
        with np.errstate(divide='ignore', invalid='ignore'):
            out = (risk[0] - risk_1_risk_2) / np.sqrt(risk_1_risk_2 * (1 - risk[1]) * (1 - risk[2]))
            return np.atleast_1d(out)

    def jacobian_g(self, risk: ndarray) -> Union[ndarray, spmatrix]:
        self._check_risk(risk)
        risk_1_risk_2 = risk[1] * risk[2]
        sqrt_factor = np.sqrt(risk_1_risk_2 * (1 - risk[1]) * (1 - risk[2]))
        # Switch off warnings since denominator may be zero
        with np.errstate(divide='ignore', invalid='ignore'):
            out = np.c_[1 / sqrt_factor,
                        0.5 * risk[2] * (1 - risk[2]) * (
                                    risk[0] * (2 * risk[1] - 1) - risk_1_risk_2) / sqrt_factor ** 3,
                        0.5 * risk[2] * (1 - risk[2]) * (
                                    risk[0] * (2 * risk[2] - 1) - risk_1_risk_2) / sqrt_factor ** 3]
            return out


class PrecisionRecallCurve(BaseMeasure):
    """Precision-Recall Curve

    Compute precision and recall for a soft classifier at different thresholds
    on the score function/probabilities.

    Parameters
    ----------
    y_score : array-like, shape=(n_instances,)
        Real-valued scores or probabilities from the classifier under
        evaluation. A larger score/probability means the classifier is more
        confident that the instance is from the positive class.

    thresholds : array-like, optional (default=None)
        A grid of thresholds at which the precision and recall will be
        computed. If None, the unique values of `y_score` are used as the
        grid of thresholds.

    pos_label : str or int, optional (default=1)
        Label used to represent the positive class.

    Notes
    -----
    A precision-recall curve can be expressed as a generalized measure by
    defining the loss function as follows:

    .. math::

        \\ell(x, y; f) = \\left[ \\mathbb{I}[f(x) > \\tau_1] \\cdot y, \\ldots, \\mathbb{I}[f(x) > \\tau_m] \\cdot y,
            \\mathbb{I}[f(x) > \\tau_1], \\ldots, \\mathbb{I}[f(x) > \\tau_m \\right]

    where :math:`[\\tau_1, \\ldots, \\tau_m]` is a grid of :math:`m`
    thresholds, :math:`f(x)` is the classifier score for instance :math:`x`,
    and :math:`\\mathbb{I}[\\cdot]` is the indicator function.
    The first :math:`m` entries of the risk
    :math:`R = \\mathbb{E}[\\ell(X, Y; f)]` correspond to the true positive
    rate of the classifier at the thresholds (in ascending order) and the
    last :math:`m` entries correspond to the predictive positive rate at
    the thresholds.

    By defining the mapping function as follows:

    .. math::

        g(R) = \\left[ \\frac{R_1}{R_{m+1}}, \\ldots, \\frac{R_m}{R_{2m}},
            \\frac{R_1}{R_1}, \\ldots, \\frac{R_m}{R_1} \\right]

    the resulting measure contains the precision at each threshold (in
    ascending order) in the first :math:`m` entries and the recall at each
    threshold (in ascending order) in the last :math:`m` entries.
    """
    def __init__(self, y_score: Union[Iterable, ndarray], thresholds: Union[Iterable, ndarray, None] = None,
                 pos_label: Union[str, int] = 1) -> None:
        super().__init__()

        self.y_score = coerce_to_1d(y_score).astype(np.floating, copy=False)
        assert_finite(self.y_score)

        if thresholds is None:
            # Set a threshold for each unique score
            self.thresholds = np.unique(self.y_score)
        else:
            # User-specified thresholds
            self.thresholds = coerce_to_1d(thresholds)

            # Check for uniqueness of thresholds
            orig_size = self.thresholds.size
            self.thresholds = np.unique(self.thresholds)
            n_duplicates = orig_size - self.thresholds.size
            if n_duplicates > 0:
                warnings.warn("`thresholds` contains {} duplicate values".format(n_duplicates))

            # Ensure thresholds contain at least some of the scores
            max_score = np.amax(self.y_score)
            min_score = np.amin(self.y_score)
            if self.thresholds[0] >= max_score or self.thresholds[-1] <= min_score:
                raise ValueError("range of `thresholds` must overlap with range of `y_score`")

            if self.thresholds[0] > min_score:
                raise ValueError("`threshold.min()` must not exceed `y_score.min()`")

        self.n_dim_risk = 2 * self.thresholds.size
        self.n_dim_g = 2 * self.thresholds.size

        self.pos_label = pos_label

    def loss(self, idx: Union[int, ndarray, Iterable], y: Union[int, float, str, ndarray, Iterable],
             x: Union[int, float, ndarray, Iterable, None] = None) -> Union[ndarray, spmatrix]:
        threshold_bin = np.digitize(self.y_score[idx], self.thresholds)
        n_instances = np.size(idx)

        # Prepare data for sparse matrix
        data = np.repeat(1.0, np.sum(threshold_bin))
        row = np.repeat(np.arange(n_instances), threshold_bin)
        col = np.repeat(threshold_bin - threshold_bin.cumsum(), threshold_bin) + np.arange(threshold_bin.sum())

        pred_positive = csr_matrix((data, (row, col)), shape=(n_instances, self.thresholds.size))
        positive = (y[:, np.newaxis] == self.pos_label) * 1
        true_positive = pred_positive.multiply(positive)

        return hstack([true_positive, pred_positive])

    def g(self, risk: ndarray) -> ndarray:
        self._check_risk(risk)

        # Extract true positive rate, predicted positive rate (for all thresholds) and positive rate
        n_thresholds = self.thresholds.size
        tpr = risk[0:n_thresholds]
        ppr = risk[n_thresholds:]
        pr = tpr[0]

        # Switch off warnings since denominator may be zero
        with np.errstate(divide='ignore', invalid='ignore'):
            # Output precision, then recall for all thresholds
            out = np.r_[tpr / ppr, tpr / pr]
            return np.atleast_1d(out)

    def jacobian_g(self, risk: ndarray) -> Union[ndarray, spmatrix]:
        self._check_risk(risk)

        # Extract true positive rate, predicted positive rate (for all thresholds) and positive rate
        n_thresholds = self.thresholds.size
        tpr = risk[0:n_thresholds]
        ppr = risk[n_thresholds:]
        pr = tpr[0]

        # Switch off warnings since denominator may be zero
        with np.errstate(divide='ignore', invalid='ignore'):
            left_upper = spdiags(1 / ppr, 0, n_thresholds, n_thresholds)
            right_upper = spdiags(-tpr / ppr ** 2, 0, n_thresholds, n_thresholds)

            # Diagonal and first column are non-zero
            rows = np.repeat(np.arange(n_thresholds), 2)
            cols = np.array(rows)
            cols[::2] = 0
            data = np.full_like(cols, 1 / pr, dtype=float)
            data[::2] = -tpr / pr ** 2
            data[0:2] = 0.
            lower_left = csr_matrix((data[1:], (rows[1:], cols[1:])), shape=(n_thresholds, n_thresholds), dtype=float)

            out = bmat([[left_upper, right_upper], [lower_left, None]])

            return out
