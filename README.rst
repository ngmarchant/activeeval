ActiveEval: Active evaluation of classifiers
============================================

ActiveEval implements a framework for active evaluation in Python. It
solves the problem of estimating the performance of a classifier on an 
unlabeled pool (test set), using labels queried from an oracle (e.g. an 
expert or crowdsourcing platform). Several methods are implemented 
including passive sampling, stratified sampling, static importance sampling 
and adaptive importance sampling. The importance sampling methods aim to 
minimize the variance of the estimated performance measure, and can yield 
more precise estimates for a given label budget. Several evaluation 
measures are currently supported including accuracy, F-measure, and 
precision-recall curves. The package is designed to be extensible.

Installation
------------
Requires Python 3.7 or higher.

**Dependencies:**

* `numpy`_
* `scipy`_
* `treelib`_

.. _numpy: https://pypi.org/project/numpy/
.. _scipy: https://pypi.org/project/scipy/
.. _treelib: https://pypi.org/project/treelib/

Install using `pip` with:

.. code-block:: bash

    $ pip install activeeval


Example
-------

.. code-block:: python

    from activeeval.measures import FMeasure, BalancedAccuracy
    from activeeval.proposals import StaticVarMin
    from activeeval.pool import Pool
    from activeeval.estimators import AISEstimator
    from activeeval import Evaluator

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    # Generate an artificial imbalanced classification dataset
    X, y = make_classification(n_samples=10000, weights=[0.99], random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    # Train a classifier to evaluate
    clf = LogisticRegression(class_weight="balanced").fit(X_train, y_train)

    # Specify pool and target evaluation measure
    pool_size = X_test.shape[0]
    pool = Pool(features=X_test)
    y_pred = clf.predict(X_test)
    fmeasure = FMeasure(y_pred)

    # Specify a static variance-minimizing proposal using the classifier to
    # estimate the oracle response
    response_est = clf.predict_proba(X_test)
    proposal = StaticVarMin(pool, fmeasure, response_est, deterministic=True)

    # Estimate the evaluation measure after collecting 1000 labels
    evaluator = Evaluator(pool, fmeasure, proposal)
    n_queries = 1000
    for _ in range(n_queries):
        # Query an instance to label
        instance_id, weight = evaluator.query()

        # Get label from oracle
        label = y_test[instance_id]

        # Update
        evaluator.update(instance_id, label, weight)

    print("Estimate of F1 score after 1000 oracle queries is", evaluator.get())

    # Reuse the samples from above to estimate a different measure
    bal_acc = BalancedAccuracy(y_pred)
    bal_acc_est = AISEstimator(bal_acc)
    for sample in evaluator.sample_history:
        bal_acc_est.update(sample.instance_id, sample.label, sample.weight)
    print("Estimate of Balanced accuracy using previous oracle queries is", bal_acc_est.get())


Support
-------
Please open an `issue <https://github.com/ngmarchant/activeeval/issues/new>`_
in this repository.

License
-------
ActiveEval is released under the MIT license.

Citation
--------

.. [1] N. G. Marchant and B. I. P. Rubinstein. (2020) "A general framework for
    label efficient online evaluation with asymptotic guarantees".
