from joblib import load
import json
import random
import sys
import os
from dependencies.model_functions import *
import numpy as np
from mutatest.mutators import *

sys.path.append(os.getcwd())

SEED = 600
random.seed(SEED)
LIMIT = 2000
# They can differ at most in a 5%
MAX_DIFF_DROPOUT = 0.05
MAX_DIFF_REPLACEMENT = 0.05


def load_test_phrases(limit=LIMIT):

    from src import p1_preprocessing

    validation_data = p1_preprocessing.read_data("data/validation.tsv")

    X_val, y_val = validation_data['title'].values, validation_data['tags'].values

    lower_limit = random.randint(0, len(X_val)-limit-1)
    upper_limit = lower_limit+limit

    return X_val[lower_limit:upper_limit], y_val[lower_limit:upper_limit]


def load_x_train():
    from src import p1_preprocessing

    return p1_preprocessing.read_data(f'data/train.tsv')['title'].values


def change_phrases_replacement(X_val, num_replacements=2, num_variants=3, selection_strategy="random"):

    mutator = ReplacementMutator(num_replacements, num_variants, selection_strategy)

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


def get_values_classifiers(X_val, y_val, X_train):

    from src import p2_text_processors

    DICT_SIZE, INDEX_TO_WORDS = p2_text_processors.get_tags_and_words(X_val, y_val)[:2]

    tfidf_vectorizer = p2_text_processors.TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2),
                                                          token_pattern='(\S+)')

    X_val_bag = p2_text_processors.get_x_values_bag(INDEX_TO_WORDS, DICT_SIZE, X_val)
    tfidf_vectorizer.fit_transform(X_train)

    X_val_tfidf = tfidf_vectorizer.transform(X_val)

    return X_val_bag, X_val_tfidf


def test_mutamorphic():

    from src import p2_text_processors
    from src import p1_preprocessing

    X_val, y_val = load_test_phrases()
    X_train = load_x_train()

    # Preprocess for both before mutating or doing other operations
    X_val = [p1_preprocessing.text_prepare(x) for x in X_val]
    X_train = [p1_preprocessing.text_prepare(x) for x in X_train]

    classifiers = load("tests/dependencies/classifiers.joblib")
    y_val = load('tests/dependencies/val_data.joblib')[:LIMIT]
    bag_classifier = classifiers["bag"]
    tfidf_classifier = classifiers["tfidf"]

    # First we ge the unaltered labels

    X_val_bag, X_val_tfidf = get_values_classifiers(X_val, y_val, X_train)

    labels_bag_og = bag_classifier.predict(X_val_bag)
    labels_tfidf_og = tfidf_classifier.predict(X_val_tfidf)

    # Now we go for the mutants
    X_val_mutates_replacement = change_phrases_replacement(X_val)

    for X_val_mutate in X_val_mutates_replacement:

        message_extra = " in replacement mutation."
        X_val_bag, X_val_tfidf = get_values_classifiers(np.array(X_val_mutate), y_val, X_train)

        labels_bag = bag_classifier.predict(X_val_bag)
        labels_tfidf = tfidf_classifier.predict(X_val_tfidf)

        check_diff(get_diff_stats(labels_bag_og, labels_bag, y_val), MAX_DIFF_REPLACEMENT, message_extra)
        check_diff(get_diff_stats(labels_tfidf_og, labels_tfidf, y_val), MAX_DIFF_REPLACEMENT, message_extra)

    # Now we go for the mutants
    X_val_mutates_dropout = change_phrases_replacement(X_val)

    for X_val_mutate in X_val_mutates_dropout:

        message_extra = " in dropout mutation."

        X_val_bag, X_val_tfidf = get_values_classifiers(np.array(X_val_mutate), y_val, X_train)

        labels_bag = bag_classifier.predict(X_val_bag)
        labels_tfidf = tfidf_classifier.predict(X_val_tfidf)

        check_diff(get_diff_stats(labels_bag_og, labels_bag, y_val), MAX_DIFF_DROPOUT, message_extra)
        check_diff(get_diff_stats(labels_tfidf_og, labels_tfidf, y_val), MAX_DIFF_DROPOUT, message_extra)

    print("All tests passed.")


test_mutamorphic()
