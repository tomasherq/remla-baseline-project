from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import json


def get_diff_stats(results_og, results_new):

    differences = {}

    # Set as a a proportion to 1
    for key in results_og:
        difference_percentage = (results_og[key]-results_new[key])/results_og[key]
        if difference_percentage < 0:
            print("YAP: ", difference_percentage)
        else:
            print("NOPE: ", difference_percentage)
        differences[key] = difference_percentage

    return differences


def check_diff(values, limit=0.20, message=""):

    # The limit can be changed by the user
    for key, difference in values.items():
        assert difference < limit, f"The value of the {key} stat, differs in {round(difference,2)*100}%{message}"
