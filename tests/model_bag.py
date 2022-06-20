from joblib import dump, load
import os
from sklearn.preprocessing import MultiLabelBinarizer
from dependencies.model_functions import *

import sys


sys.path.append(os.getcwd())

# This test needs to run the preprocessing first!!


def test_model_bag():

    from src import p3_train
    from src import p4_predict
    from src import p5_evaluation

    # Get the data obtained from a previous run
    text_process_data = load(f'tests/dependencies/models_data.joblib')

    bag_of_words_data = text_process_data["bag"]

    y_val = bag_of_words_data["y_val"]
    y_train = bag_of_words_data["y_train"]
    tags_counts = bag_of_words_data["tags_counts"]

    # Train the classifier
    mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
    y_train = mlb.fit_transform(y_train)
    y_val = mlb.fit_transform(y_val)

    # Get the models with two different seeds
    bag_classifier_1 = p3_train.train_bag(bag_of_words_data, y_train, 2801)
    bag_classifier_2 = p3_train.train_bag(bag_of_words_data, y_train, 1998)

    # Retrieve the validation data
    X_val_bag = text_process_data["bag"]["X_val"]

    # Run the predictions
    predictions_1 = p4_predict.run_prediction(bag_classifier_1, X_val_bag)
    predictions_2 = p4_predict.run_prediction(bag_classifier_2, X_val_bag)

    # See if the difference in accuracy or any stat is less than a threshold
    return get_diff_stats(p5_evaluation.get_eval_results(y_val, predictions_1["labels"], predictions_1["scores"]),
                          p5_evaluation.get_eval_results(y_val, predictions_2["labels"], predictions_2["scores"]))


check_diff(test_model_bag())
