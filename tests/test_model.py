import os
import subprocess
import sys

import joblib
from sklearn.datasets import load_iris


def test_model_trains_and_predicts():
    subprocess.run([sys.executable, "train.py"], check=True)
    assert os.path.exists("model.joblib")
    model = joblib.load("model.joblib")
    X, y = load_iris(return_X_y=True)
    assert (model.predict(X) == y).mean() > 0.9
