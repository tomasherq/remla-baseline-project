# Multilabel classification on Stack Overflow tags
Predict tags for posts from StackOverflow with multilabel classification approach.

## Dataset
- Dataset of post titles from StackOverflow

## Transforming text to a vector
- Transformed text data to numeric vectors using bag-of-words and TF-IDF.

## MultiLabel classifier
[MultiLabelBinarizer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html) to transform labels in a binary form and the prediction will be a mask of 0s and 1s.

[Logistic Regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) for Multilabel classification
- Coefficient = 10
- L2-regularization technique

## Evaluation
Results evaluated using several classification metrics:
- [Accuracy](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
- [F1-score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
- [Area under ROC-curve](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
- [Area under precision-recall curve](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score)

## Libraries
- [Numpy](http://www.numpy.org/) — a package for scientific computing.
- [Pandas](https://pandas.pydata.org/) — a library providing high-performance, easy-to-use data structures and data analysis tools for the Python
- [scikit-learn](http://scikit-learn.org/stable/index.html) — a tool for data mining and data analysis.
- [NLTK](http://www.nltk.org/) — a platform to work with natural language.

Note: this sample project was originally created by @partoftheorigin


## Executing the pipeline

To execute the pipeline we have to first build and run the docker image of this repository:

```
docker build --progress plain . -t remla-project-g18
docker run -it --rm -v "$(pwd)":/root/project remla-project-g18

```

Once inside docker we have to access the project folder and we can execute the pipeline:

```
cd project
dvc init
dvc repro
```

## Results obtained after the execution

The results of our execution will be reported in a PDF file whose location will be indicated at the end of the pipeline execution. These results include the resources used during the pipeline execution, the run times for the different stages of the pipeline, the size of the input files used for the model and the metrics obtained for the different models.

The output files are the following:

- results/metrics-(bag or tfidf).json: Results for the different metrics measured for the model. 
- monitoring/metrics/runtimes/runtimes.txt: Run times of the different pipeline stages.
- monitoring/metrics/results_execution.json: Resources used during the pipeline execution by the computer.
- static_analysis/report_lynter/report.txt: Output of running `pylint src`.
- run_report.pdf: Summary of the results obtained.
- results/popular_words_(bag or tfidf).json: Most and least relevant words for each text transformer.



