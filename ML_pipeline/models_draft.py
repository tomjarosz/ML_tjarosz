import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
import json
import numpy as np
import pylab as pl
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn import metrics, svm

MODELS = ['LR']
# ['RF', 'LR', 'SVM', 'GB', 'DT', 'KNN']

CLASSIFIERS = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
               'LR': LogisticRegression(penalty='l1', C=1e5),
               'SVM': svm.LinearSVC(random_state=0, dual=False),
               'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
               'DT': DecisionTreeClassifier(),
               'KNN': KNeighborsClassifier(n_neighbors=3)}

PARAMETERS = {'RF': {'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
              'LR': {'penalty': ['l1','l2'], 'C': [0.0001,0.01,0.1,1,10]},
              'SVM':{'C' :[0.0001,0.01,0.1,1,10], 'penalty': ['l1', 'l2']},
              'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
              'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
              'KNN':{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}}

def prep_model_data(df_train, dep_vars, indep_vars):
    '''
    Prepare data for evaluating various models, using the 
    training dataset, list of indepedent variables, and 
    dependent variables.

    Utilizes train_test_split function for parse data
    '''
    y_train = df_train[dep_vars]
    y_train = np.ravel(y_train)
    x_train = df_train[indep_vars]

    x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = .2, random_state = 0)
    return(x_train, x_validate, y_train, y_validate)

def fit_model(x_train, x_validate, y_train, y_validate, MODELS, CLASSIFIERS, PARAMETERS):
    '''
    Fits and predicts a model for the data using
    the parsed data from the train_test_split function
    '''
    with open('model_comparison.csv', 'wb') as csvfile:
    table = csv.writer(csvfile, delimiter = ',')
    table.writerow([MODEL, PARAMETERS, ACCURACY, PRECISION, RECALL, F1, AUC])

    fit_models = []
    for index, CLASSIFIERS in enumerate([CLASSIFIERS[x] for x in MODELS]):
        values = PARAMETERS[MODELS[index]]
        for param in ParameterGrid(values):
            CLASSIFIERS.set_params(**param)
            CLASSIFIERS.fit(x_train, y_train)
            pred_probs = CLASSIFIERS.predict(x_validate)        
            eval_model = evaluate(y_validate, pred_probs)
            fit_models.append(eval_model)   
            print(eval_model)
            table.writerow([x, param] + [eval_model])


    
    # # return fit_models
    # print(fit_models)      

def evaluate(y_validate, pred_probs):
    '''
    Evaluates the fit of the model on the validation data 
    using an accuracy metric.
    '''
    
    ACCURACY = metrics.accuracy_score(y_validate, pred_probs)
    PRECISION = metrics.precision_score(y_validate, pred_probs)
    RECALL = metrics.recall_score(y_validate, pred_probs)
    F1 = metrics.f1_score(y_validate, pred_probs)
    AUC = metrics.roc_auc_score(y_validate, pred_probs)
    # PRCurves = 

    eval_metrics = [ACCURACY, PRECISION, RECALL, F1, AUC] #, PRCurves]
    return eval_metrics
# def score_data(logit_model, df_test, dep_vars, indep_vars):
#     '''
#     Utilizes the best model to predict outcomes in the 
#     testing dataset
#     '''
#     x_test = df_test[indep_vars]
#     y_test = logit_model.predict_proba(x_test)

#     prob_list = []
#     for prob in y_test:
#         prob_list.append(prob[1])

#     df_test[dep_vars] = prob_list
#     return(df_test[dep_vars])