from joblib import load
import json
import random
import sys
import os
import numpy as np
from mutatest.mutators import *

sys.path.append(os.getcwd())


SEED = 600
random.seed(SEED)
LIMIT = 2000
# Difference as proportion of 1
MAX_DIFF_DROPOUT = 0.50
MAX_DIFF_REPLACEMENT = 0.20


def load_classifiers_data(filename):
    from src import p1_preprocessing

    data = p1_preprocessing.read_data(filename)

    X, y = data['title'].values, data['tags'].values

    return X, y


def load_test_phrases():

    X, y = load_classifiers_data("data/validation.tsv")

    return X[:LIMIT], y


def load_train_phrases():

    return load_classifiers_data("data/train.tsv")


def change_phrases_replacement(X_val, num_replacements=2, num_variants=3, selection_strategy="most_common_first"):

    mutator = ReplacementMutator(num_replacements, num_variants, selection_strategy)

    new_x_vals = []
    for i in range(0, num_variants):
        new_x_vals.append(list())
    counter = 0
    for sentence in X_val:
        results = mutator.mutate(sentence, SEED)

        for i in range(0, num_variants):

            if len(results) == num_variants:
                new_x_vals[i].append(results[i])
            else:
                # This way we ENSURE that both results have equal length
                new_x_vals[i].append(sentence)

    return new_x_vals
    # To be implemented


def change_phrases_dropout(X_val, num_replacements=2, num_variants=2):

    mutator = DropoutMutator(num_replacements, num_variants)

    new_x_vals = []
    for i in range(0, num_variants):
        new_x_vals.append(list())

    for sentence in X_val:
        results = mutator.mutate(sentence, SEED)

        for i in range(0, num_variants):

            if len(results) == num_variants:
                new_x_vals[i].append(results[i])
            else:
                # This way we ENSURE that both results have equal length
                new_x_vals[i].append(sentence)

    return new_x_vals


def get_values_classifiers(X_val, y_val, X_train, y_train):

    from src import p2_text_processors

    DICT_SIZE, INDEX_TO_WORDS, WORDS_TO_INDEX, ALL_WORDS, tags_counts = p2_text_processors.get_tags_and_words(
        X_train, y_train)

    X_val_bag = p2_text_processors.get_x_values_bag(WORDS_TO_INDEX, DICT_SIZE, X_val)

    X_val_bag = p2_text_processors.get_x_values_bag(WORDS_TO_INDEX, DICT_SIZE, X_val)

    tfidf_vectorizer = p2_text_processors.TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2),
                                                          token_pattern='(\S+)')
    tfidf_vectorizer.fit_transform(X_train)

    X_val_tfidf = tfidf_vectorizer.transform(X_val)

    return X_val_bag, X_val_tfidf


def get_results_predictions(classifier, X_val, y_val):
    from src import p4_predict
    from src import p5_evaluation

    predictions = p4_predict.run_prediction(classifier, X_val)

    return p5_evaluation.get_eval_results(y_val, predictions["labels"], predictions["scores"])


def test_mutamorphic():

    # from src import p2_text_processors
    from dependencies.model_functions import check_diff, get_diff_stats
    from src import p1_preprocessing

    X_val, y_val = load_test_phrases()
    X_train, y_train = load_train_phrases()

    # Preprocess for both before mutating or doing other operations
    X_val = [p1_preprocessing.text_prepare(x) for x in X_val]
    X_train = [p1_preprocessing.text_prepare(x) for x in X_train]

    # This y_val has been treated by the multilabel classifier
    y_val = load('tests/dependencies/val_data.joblib')[:LIMIT]

    classifiers = load("tests/dependencies/classifiers.joblib")
    bag_classifier = classifiers["bag"]
    tfidf_classifier = classifiers["tfidf"]

    # We obtain the metrics for the unaltered data

    X_val_bag, X_val_tfidf = get_values_classifiers(X_val, y_val, X_train, y_train)

    results_bag_og = get_results_predictions(bag_classifier, X_val_bag, y_val)
    results_tfidf_og = get_results_predictions(tfidf_classifier, X_val_tfidf, y_val)

    # We compare the senteces obtained by using the replacement mutator
    X_val_mutates_replacement = change_phrases_replacement(X_val)

    for X_val_mutate in X_val_mutates_replacement:

        X_val_bag, X_val_tfidf = get_values_classifiers(np.array(X_val_mutate), y_val, X_train, y_train)

        results_bag_mutated = get_results_predictions(bag_classifier, X_val_bag, y_val)
        results_tfidf_mutated = get_results_predictions(tfidf_classifier, X_val_tfidf, y_val)

        check_diff(get_diff_stats(results_bag_og, results_bag_mutated),
                   MAX_DIFF_REPLACEMENT, " BOW in replacement mutation.")

        check_diff(get_diff_stats(results_tfidf_og, results_tfidf_mutated),
                   MAX_DIFF_REPLACEMENT, " TFIDF in replacement mutation.")

    # We compare the senteces obtained by using the replacement mutator

    X_val_mutates_dropout = change_phrases_dropout(X_val)

    for X_val_mutate in X_val_mutates_dropout:

        X_val_bag, X_val_tfidf = get_values_classifiers(np.array(X_val_mutate), y_val, X_train, y_train)

        results_bag_mutated = get_results_predictions(bag_classifier, X_val_bag, y_val)
        results_tfidf_mutated = get_results_predictions(tfidf_classifier, X_val_tfidf, y_val)

        check_diff(get_diff_stats(results_bag_og, results_bag_mutated),
                   MAX_DIFF_DROPOUT, " BOW in dropout mutation.")

        check_diff(get_diff_stats(results_tfidf_og, results_tfidf_mutated),
                   MAX_DIFF_DROPOUT, " TFIDF in dropout mutation.")

    print("All tests passed.")


test_mutamorphic()
