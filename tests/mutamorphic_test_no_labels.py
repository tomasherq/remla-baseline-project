from mutatest.test_runner import MutamorphicTest, MutamorphicTestCase
from mutatest.mutators import ReplacementMutator, DropoutMutator

from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import mean_squared_error
from joblib import dump, load
import numpy as np
import pandas as pd
import sys
import os
from random import Random
from typing import List

sys.path.append(os.getcwd())


WORDS_TO_INDEX = load('tests/dependencies/words_to_index.joblib')
DICT_SIZE = len(WORDS_TO_INDEX)
CLASSIFIERS = load('tests/dependencies/classifiers.joblib')
TEST_DATA = pd.read_csv(f'data/test.tsv', sep='\t', dtype={"title": object})[["title"]].values.tolist()
TEST_DATA = [x[0] for x in TEST_DATA]

SEED = 600
NUMBER_OF_TEST_CASES = 1000
MAXIMUM_DISTANCE = 20.0


def get_input_sentences(rng: Random) -> List[str]:
    return rng.sample(TEST_DATA, NUMBER_OF_TEST_CASES)


def run_bag_of_words(sentence: str) -> np.ndarray:

    from src.p1_preprocessing import text_prepare
    from src.p2_text_processors import bag_of_words
    classifier: OneVsRestClassifier = CLASSIFIERS['bag']
    cleaned_input = text_prepare(sentence)
    input_vector = bag_of_words(cleaned_input, WORDS_TO_INDEX, DICT_SIZE)
    input_vector = np.expand_dims(input_vector, 0)

    scores = classifier.decision_function(input_vector)

    return scores


def distance_function(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    return mean_squared_error(vector_a, vector_b)


def test_mutamorphic_bow_replacement():
    rng = Random()
    rng.seed(SEED)
    input_sentences = get_input_sentences(rng)
    test: MutamorphicTest[np.ndarray] = MutamorphicTest(
        input_sentences,
        ReplacementMutator(num_replacements=1, num_variants=5),
        run_bag_of_words,
        distance_function,
        random_seed=SEED
    )
    test.run()

    dump(test, "tests/dependencies/test_object_replacement50.joblib")

    print(test.average_similarity)
    assert test.average_similarity < MAXIMUM_DISTANCE, f"The average similarity is {test.average_similarity} for the replacement."


def test_mutamorphic_bow_dropout():
    rng = Random()
    rng.seed(SEED)
    input_sentences = get_input_sentences(rng)
    test: MutamorphicTest[np.ndarray] = MutamorphicTest(
        input_sentences,
        DropoutMutator(num_dropouts=1, num_variants=5),
        run_bag_of_words,
        distance_function,
        random_seed=SEED
    )
    test.run()

    dump(test, "tests/dependencies/test_object_dropout.joblib")

    assert test.average_similarity < MAXIMUM_DISTANCE, f"The average similarity is {test.average_similarity} for the dropout."


if __name__ == "__main__":

    test_mutamorphic_bow_replacement()
    test_mutamorphic_bow_dropout()
