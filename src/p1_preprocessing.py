
import pandas as pd
from ast import literal_eval
from nltk.corpus import stopwords
import nltk
import re
import argparse
import sys
import os
from joblib import dump
from mutatest.mutators import *


nltk.download('stopwords')
sys.path.append(os.getcwd())
selected_options = {}


def check_arguments():

    parser = argparse.ArgumentParser(
        description='Extra arguments can be specified for having a mutamorphic transformation.')

    parser.add_argument("--replace", "-r", nargs=2, metavar=('<int>', '<str>'),
                        help="This option activates transformation by replacement. Need to specify the following: <Nº of words to replace> <Type of replace (random/most_common_first)>", required=False)

    parser.add_argument("--drop", "-d", type=int,
                        help="This option activates transformation by dropout. Need to specify the number of words to drop.", required=False)
    parser.add_argument("--jumps", "-j",  type=int, required=False, help="Change every 1/jump words.")

    args = parser.parse_args()

    selected_options = {}

    replace_valid = args.replace is not None and len(args.replace) > 1

    if replace_valid:
        if args.replace[1] in ["random", "most_common_first"] and args.replace[0].isdigit():

            print("Selected replace")
            words_replace = int(args.replace[0])
            if words_replace > 0:
                selected_options["replace"] = {"n_words_replace": words_replace, "strategy": args.replace[1]}

    elif args.drop is not None:
        print("Selected drop")
        selected_options["drop"] = args.drop

    selected_options["jumps"] = {"number": args.jumps, "counter": 0}
    return selected_options


# RegEx expressions to clean the data
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

# The arguments for having mutamorphic stuff


def read_data(filename):
    data = pd.read_csv(filename, sep='\t', dtype={"title": object, "tags": object})[["title", "tags"]]

    data['tags'] = data['tags'].apply(literal_eval)

    return data


def text_prepare(text, mutator=None):
    """
        text: a string
        return: modified initial string
    """

    text = text.lower()  # lowercase text
    text = re.sub(REPLACE_BY_SPACE_RE, " ", text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(BAD_SYMBOLS_RE, "", text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = " ".join([word for word in text.split() if not word in STOPWORDS])  # delete stopwords from text

    # We manipulate the text after it has been processed
    if mutator is not None:

        if selected_options["jumps"]["counter"] == 0:
            mutant_text = mutator.mutate(text, 36)
            if len(mutant_text) > 0:
                text = mutant_text[0]

        if selected_options["jumps"]["number"] is not None:
            selected_options["jumps"]["counter"] += 1
            if selected_options["jumps"]["number"] <= selected_options["jumps"]["counter"]:
                selected_options["jumps"]["counter"] = 0

    return text


def get_preprocessed_data(path_data="data/"):

    # Think that we have to change this please!

    mutator = None
    if "replace" in selected_options:
        mutator = ReplacementMutator(selected_options["replace"]["n_words_replace"],
                                     1, selected_options["replace"]["strategy"])

    elif "dropout" in selected_options:
        mutator = DropoutMutator(selected_options["drop"], 1)

    # Read the data to be used in the project
    train = read_data(f'{path_data}train.tsv')
    validation = read_data(f'{path_data}validation.tsv')
    test = pd.read_csv(f'{path_data}test.tsv', sep='\t', dtype={"title": object})[["title"]]

    # Separate trainning and validation
    X_train, y_train = train['title'].values, train['tags'].values
    X_val, y_val = validation['title'].values, validation['tags'].values
    X_test = test['title'].values

    # Retrieve preprocesed data
    X_train = [text_prepare(x, mutator) for x in X_train]
    X_val = [text_prepare(x) for x in X_val]
    X_test = [text_prepare(x) for x in X_test]

    return {"X_train": X_train, "X_val": X_val, "X_test": X_test, "y_train": y_train, "y_val": y_val}


def main():
    preprocessed_data = get_preprocessed_data()

    dump(preprocessed_data, 'output/preprocessed_data.joblib')


if __name__ == "__main__":
    from monitoring.monitoring_tools import start_execution, register_timestamp

    start_execution(sys.argv[0])
    selected_options = check_arguments()
    main()
    register_timestamp(sys.argv[0], "end")
