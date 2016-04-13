import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
import json
import numpy as np
import pylab as pl
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics

def prep_model_data(df_train, dep_vars, indep_vars):
    y_train = df_train[dep_vars]
    y_train = np.ravel(y_train)
    x_train = df_train[indep_vars]

    x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = .2, random_state = 0)
    return(x_train, x_validate, y_train, y_validate)

def logit_model(x_train, x_validate, y_train, y_validate):
    #could pass in a list of models and loop through them to fit and predict probs
    logit_model = LogisticRegression()
    logit_model.fit(x_train, y_train)
    pred_probs = logit_model.predict(x_validate)
    return pred_probs, logit_model  

def evaluate(y_validate, pred_probs):
    accuracy = metrics.accuracy_score(y_validate, pred_probs)
    return accuracy

def score_data(logit_model, df_test, dep_vars, indep_vars):
    
    x_test = df_test[indep_vars]
    y_test = logit_model.predict_proba(x_test)

    prob_list = []
    for prob in y_test:
        prob_list.append(prob[1])

    df_test[dep_vars] = prob_list
    return(df_test[dep_vars])