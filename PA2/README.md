Assignment 2: Machine Learning Pipeline

The goal of this assignment is to build a simple, modular, extensible, machine learning pipeline in Python. The pipeline should have functions that can do the following tasks:

1. Read Data: For this assignment, assume input is CSV
2. Explore Data: You can use the code you wrote for assignment 1 here to generate distributions and data summaries
3. Pre-Process Data: Fill in misssing values
4. Generate Features: Write a sample function that can discretize a continuous variable and one function that can take a categorical variable and create binary variables from it.
5. Build Classifier: For this assignment, select any classifer you feel comfortable with (Logistic Regression for example)
6. Evaluate Classifier: you can use any metric you choose for this assignment (accuracy is the easiest one)

Assignment: The outputs for this assignment are:

1. Code for the ML Pipeline with the components listed above

2. Results of running the code on the following problem:

The data can be downloaded from https://www.kaggle.com/c/GiveMeSomeCredit/data

Your task is to train one or more models on the training data and generate delinquency scores for the test data. Don't worry too much about the scores being good at this point. JUst try running some of the models we covered in class last week.

There is useful code in ipython notebooks in this git repo: https://github.com/yhat/DataGotham2013/

The purpose of this homework is to start building your ML pipeline. Don't worry too much about solving the specific problem well. Since we haven't covered evaluation yet, don't worry about creating validation sets or cross-validation. You can evaluate on the training set for now to pick a good model and then use that to score the new data in the cs-test.csv