# Execution report

This report provides of a brief summary of how the execution of the pipeline went.

## Resources used

The CPU, memory and disk usage during the execution of the pipeline are reflected in the following graphs:

<center>
![Alt text](##LocationGraph## "Resources used during execution.")
</center>
The average CPU, memory and disk usages during the execution are the following:

<center>

| Resource    | Average use (%)        |
|-------------|------------------------|
| CPU         | ##cpu_usage##   |
| RAM memory  | ##ram_memory## |
| Swap memory | ##swap_memory##           |
| Disk        | ##disk_usage##         |

</center>

The total number of reads and writes to the disk during the execution are the following:

<center>

| I/O action | Total done |
|------------|------------|
| Reads      | ##reads##  |
| Writes     | ##writes## |

</center>

## Pipeline run time

The run times for the different stages (in seconds) are the following:
<center>

| Stage                  | Run time (s)            |
|------------------------|-------------------------|
| Preprocessing          | ##p1_preprocessing##    |
| Obtain text processors | ##p2_text_processors##  |
| Train the models       | ##p3_train##            |
| Run the predictions    | ##p4_predict##          |
| Evaluate the results   | ##p5_evaluation##       |
| Analyze the features   | ##p6_analyze_features## |
| Total time needed      | ##total##               |

</center>

<div style="page-break-after: always;"></div>

## Size of the data used

Size of the train, validation and test datasets used in bytes.

<center>

| Data set  | Size (MB)         | 
|-----------|-------------------|
| Train     | ##train##         | 
| Validation| ##validation##    |
| Test      | ##test##          |

</center>


## Metrics obtained for the models

Values for the different metrics obtained for the two models tested.


<center>


| Metric    | Bag of words     | TF-IDF             |
|-----------|------------------|--------------------|
| Accuracy  | ##accuracybag##  | ##accuracytfidf##  |
| F1 Score  | ##f1bag##        | ##f1tfidf##        |
| Precision | ##precisionbag## | ##precisiontfidf## |
| ROC       | ##rocbag##       | ##roctfidf##       |

</center>


## Feature analisis

The model that performed the best according to most metrics is the ##bestmodel##.

The most seen words in Bag of Words were the following:

##Bag_words_most##

and the least seen were:

##Bag_words_least##

For the TF-IDF model, the top positive words were:

##TFIDF_words_most##

and the top negative were:

##TFIDF_words_least##


