import pytest
# TODO: add necessary import
from ml.model import train_model, compute_model_metrics, inference
import os
import sys
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

root_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(root_dir)

# TODO: implement the first test. Change the function name and input as needed
def test_train_model():
    """
    Test pipeline of training model
    """
    X = np.random.rand(20, 5)
    y = np.random.randint(2, size=20)
    model = train_model(X, y)
    assert isinstance(model, BaseEstimator) and isinstance(
        model, ClassifierMixin)



# TODO: implement the second test. Change the function name and input as needed
def test_inference_model():
    """
      Test inference of model
      """
    X_train = np.random.rand(60, 5)
    y_train = np.random.randint(2, size=60)
    random_forest = train_model(X_train, y_train)
    y_preds = inference(random_forest, X_train)

    assert y_train.shape == y_preds.shape


# TODO: implement the third test. Change the function name and input as needed
def test_compute_model_metrics():
    """
    Test compute_model_metrics
    """
    y_true, y_preds = [1, 1, 0], [0, 1, 1]
    precision, recall, fbeta = compute_model_metrics(y_true, y_preds)
    assert precision is not None
    assert recall is not None
    assert fbeta is not None
