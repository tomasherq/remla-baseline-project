from dependencies.model_functions import *
from joblib import dump, load
import os
import sys
from sklearn.preprocessing import MultiLabelBinarizer


sys.path.append(os.getcwd())


def test_model_tfidf():

    from src import p3_train
    from src import p4_predict
    from src import p5_evaluation

    # Get the data obtained from a previous run
    text_process_data = load(f'tests/dependencies/models_data.joblib')

    tfidf_data = text_process_data["tfidf"]

    y_val = text_process_data["bag"]["y_val"]
    y_train = text_process_data["bag"]["y_train"]
    tags_counts = text_process_data["bag"]["tags_counts"]

    # Train the classifier
    mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
    y_train = mlb.fit_transform(y_train)
    y_val = mlb.fit_transform(y_val)

    # Get the models
    tfidf_classifier_1 = p3_train.train_tfidf(tfidf_data, y_train, 2801)
    tfidf_classifier_2 = p3_train.train_tfidf(tfidf_data, y_train, 1998)

    # Retrieve the validation data
    X_val_tfdif = tfidf_data["X_val"]

    # Run the predictions
    predictions_1 = p4_predict.run_prediction(tfidf_classifier_1, X_val_tfdif)
    predictions_2 = p4_predict.run_prediction(tfidf_classifier_2, X_val_tfdif)

    # See if the difference in accuracy or any stat is less than a threshold
    return get_diff_stats(p5_evaluation.get_eval_results(y_val, predictions_1["labels"], predictions_1["scores"]),
                          p5_evaluation.get_eval_results(y_val, predictions_2["labels"], predictions_2["scores"]))


check_diff(test_model_tfidf())
