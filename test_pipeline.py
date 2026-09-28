import os

import pytest

from main import run_prepare, run_train, DATA_PATH


@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="CSV manquant")
def test_prepare_and_train():
    x_train, x_test, y_train, y_test = run_prepare()
    assert len(x_train) > len(x_test)
    assert len(x_train) == len(y_train)
    model = run_train(x_train, y_train)
    assert hasattr(model, "predict")
