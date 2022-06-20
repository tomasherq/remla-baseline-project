
import sys
import os
from joblib import load, dump
sys.path.append(os.getcwd())


def test_text_prepare():

    from src.p1_preprocessing import text_prepare

    # Load the data obtained from a previous successful run
    data_for_test = load("tests/dependencies/preprocessing_data.joblib")

    # Check that the current preprocessing gives the same output
    for ex, ans in zip(data_for_test["input"], data_for_test["expected_output"]):
        if text_prepare(ex) != ans:
            return False
    return True


assert test_text_prepare(), "Preprocessing is not working correctly"
