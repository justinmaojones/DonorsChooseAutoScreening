Welcome to the DonorsChooseAutoScreening wiki!

[See the Wiki here](https://github.com/justinmaojones/DonorsChooseAutoScreening/wiki/02.-Overview)

Team Members: Justin Mao-Jones, Charles Guthrie, Yasumasa Miyamoto, Lucy Wang

#Summary

In a nutshell, the aim of this project was to build a tool that could automatically screen new submissions to DonorsChoose.org.  Submissions are textual, consisting primarily of "essays" and itemized lists of requested resources (e.g. new books, calculators, ipads, etc). More specifically, this was a binary classification of essays and itemized lists associated with each submission.  We used NLP, [TF-IDF](https://github.com/justinmaojones/DonorsChooseAutoScreening/wiki/8.-TF-IDF-Interpretation), and other features to train the model.

The purpose of this wiki is to provide an overview of the modeling and analysis performed.  The material in this wiki goes beyond what is in a [final report](https://github.com/justinmaojones/DonorsChooseAutoScreening/blob/master/FinalReport_DonorsChooseAutoScreening.pdf) produced as part of a [class project](https://github.com/charlesdguthrie/frontrow/blob/master/final_report/fall14_ds_project_instructions.pdf).  Also, please feel free to browse our [code](https://github.com/justinmaojones/DonorsChooseAutoScreening).

#To Run the Code
You will need the following data sets in the 'data' directory:
- opendata_projects.csv (found at http://data.donorschoose.org/open-data/overview/)
- all_essays.csv
- essays_and_labels.csv

Package requirements:
- pandas
- textmining
- prettyplotlib


To build the models:
1. download the data [to where?]
1. run main.py to process the data and run the models

Disclaimer: This model has not been incorporated at DonorsChoose.org.
