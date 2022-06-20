from joblib import dump, load
from mutamorphic_test_no_labels import run_bag_of_words, distance_function
from matplotlib import pyplot as plt
import numpy as np
import math
from typing import Tuple, Set, Dict
from os import path

TEST_CASES_FILE_REPL = "grouped_test_cases_repl.joblib"
TEST_CASES_FILE_DROP = "grouped_test_cases_drop.joblib"
BOXPLOT_FILE_REPL = "boxplot_values_repl.joblib"
BOXPLOT_FILE_DROP = "boxplot_values_drop.joblib"


def group_test_cases(test_cases, num_bins, bin_start, bin_end):
    grouped = []
    for i in range(num_bins):
        grouped.append(set())
    outliers = set()
    for i, x in enumerate(test_cases):
        if i % 100 == 0:
            print(f"grouping #{i}")
        s = x.average_similarity
        index = (s - bin_start) / (bin_end - bin_start) * num_bins
        index = math.floor(index)
        if 0 <= index < num_bins:
            grouped[index].add(x)
        else:
            outliers.add(x)

    return grouped, outliers


if __name__ == "__main__":

    test_repl = load("test_object_replacement.joblib")
    test_drop = load("test_object_dropout.joblib")

    filtered_repl = [x for x in test_repl.test_cases if len(x.similarities) > 0]
    filtered_drop = [x for x in test_drop.test_cases if len(x.similarities) > 0]

    averages_repl = [x.average_similarity for x in filtered_repl]
    averages_drop = [x.average_similarity for x in filtered_drop]

    num_bins = 12
    bin_start = 0
    bin_end = 6.5
    bins = np.arange(bin_start, bin_end, step=0.5).tolist()

    if path.exists(TEST_CASES_FILE_REPL):
        _temp = load(TEST_CASES_FILE_REPL)
        test_cases_grouped_repl = _temp["grouped"]
        test_cases_outliers_repl = _temp["outliers"]
    else:
        test_cases_grouped_repl, test_cases_outliers_repl =\
            group_test_cases(filtered_repl, num_bins, bin_start, bin_end)
        dump({"grouped": test_cases_grouped_repl,
              "outliers": test_cases_outliers_repl},
             TEST_CASES_FILE_REPL)

    if path.exists(TEST_CASES_FILE_DROP):
        _temp = load(TEST_CASES_FILE_DROP)
        test_cases_grouped_drop = _temp["grouped"]
        test_cases_outliers_drop = _temp["outliers"]
    else:
        test_cases_grouped_drop, test_cases_outliers_drop =\
            group_test_cases(filtered_drop, num_bins, bin_start, bin_end)
        dump({"grouped": test_cases_grouped_drop,
              "outliers": test_cases_outliers_drop},
             TEST_CASES_FILE_DROP)


    plt.hist(averages_repl, bins=bins)
    plt.title("Histogram of average output similarity for\nthe replacement mutator no-label mutamorphic test")
    plt.xlabel("average test case similarity (using MSE)")
    plt.ylabel("number of occurrences")
    plt.show()

    plt.figure()
    plt.hist(averages_drop, bins=np.arange(6.5, step=0.5).tolist())
    plt.title("Histogram of average output similarity for\nthe dropout mutator no-label mutamorphic test")
    plt.xlabel("average test case similarity (using MSE)")
    plt.ylabel("number of occurrences")
    plt.show()

    boxplot_labels = []
    for i in range(num_bins):
        # boxplot_labels.append(f"{(bins[i] + bins[i+1])/2}")
        boxplot_labels.append(f"{bins[i]}")
    if path.exists(BOXPLOT_FILE_REPL):
        boxplot_values_repl = load(BOXPLOT_FILE_REPL)
    else:
        boxplot_values_repl = []
        for i, group in enumerate(test_cases_grouped_repl):
            if i % 100 == 0:
                print(f"box plot values repl #{i}")
            boxplot_values_repl.append([len(x.get_nontrivial_words()) for x in group])
        dump(boxplot_values_repl, BOXPLOT_FILE_REPL)

    if path.exists(BOXPLOT_FILE_DROP):
        boxplot_values_drop = load(BOXPLOT_FILE_DROP)
    else:
        boxplot_values_drop = []
        for i, group in enumerate(test_cases_grouped_drop):
            if i % 100 == 0:
                print(f"box plot values drop #{i}")
            boxplot_values_drop.append([len(x.get_non_stopwords()) for x in group])
        dump(boxplot_values_drop, BOXPLOT_FILE_DROP)


    plt.figure()
    plt.boxplot(boxplot_values_repl, labels=boxplot_labels)
    plt.title("Boxplot of number of nontrivial words per test case for each histo-\ngram bin for the replacement mutator no-label mutamorphic test")
    plt.xlabel("left bin edges from corresponding histogram (average test case similarity)")
    plt.ylabel("number of nontrivial words")
    plt.show()

    plt.figure()
    plt.boxplot(boxplot_values_drop, labels=boxplot_labels)
    plt.title("Boxplot of number of non stopwords per test case for each\nhistogram bin for the dropout mutator no-label mutamorphic test")
    plt.xlabel("left bin edges from corresponding histogram (average test case similarity)")
    plt.ylabel("number of non-stopwords")
    plt.show()

    # num_nontrivial_repl = [len(x.get_nontrivial_words()) for x in filtered_repl]
    # plt.figure()
    # plt.scatter(num_nontrivial_repl, averages_repl)
    # plt.xlabel("number of nontrivial words")
    # plt.ylabel("average similarity of test case")
    # plt.show()



    a=3