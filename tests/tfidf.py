import sys
import os
import json
from joblib import load
sys.path.append(os.getcwd())


def test_tfidf():

    from src import p2_text_processors

    # Load the data used for the test
    validation_data = load("tests/dependencies/tfidf_process_data.joblib")

    train_values = validation_data["train_values"]
    result_vocab = validation_data["result_vocab"]

    # Get the tfidf vocabulary with the train values
    tfidf_vocabulary = p2_text_processors.tfidf_features(
        train_values["train"], train_values["val"], train_values["test"])[3]

    # Check that is equal to the obtained in a successful run.
    return tfidf_vocabulary == result_vocab


assert test_tfidf(), "TFIDF is not working properly"
