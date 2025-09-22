import os
import pytest
from model import predict_image

def test_model_file_exists():
    """Check that the trained model file exists."""
    assert os.path.exists("models/mnist_cnn.pth")

def test_prediction_runs():
    """Check that prediction function returns an int digit between 0–9."""
    test_img = "app/static/uploads/test_digit.png"
    # if no test image exists, just skip
    if not os.path.exists(test_img):
        pytest.skip("No test image available")
    pred = predict_image(test_img)
    assert isinstance(pred, int)
    assert 0 <= pred <= 9
