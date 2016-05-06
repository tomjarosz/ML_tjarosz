import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
import json
import numpy as np
import pylab as pl
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn import metrics, svm
from sklearn.metrics import precision_recall_curve

#INCLUDES CODE FROM RAYID GHANI'S MAGICLOOP.PY

MODELS = ['RF', 'LR', 'SVM', 'AB', 'DT', 'KNN']

CLASSIFIERS = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
               'LR': LogisticRegression(penalty='l1', C=1e5),
               'SVM': svm.LinearSVC(random_state=0, dual=False),
               'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
               'DT': DecisionTreeClassifier(),
               'KNN': KNeighborsClassifier(n_neighbors=3)}

PARAMETERS = {'RF': {'n_estimators': [1,10,100], 'max_depth': [1,5,10], 'max_features': ['sqrt','log2'],'min_samples_split': [5,10]},
              'LR': {'penalty': ['l1','l2'], 'C': [0.0001,0.01,0.1,1,10]},
              'SVM':{'C' :[0.0001,0.01,0.1,1,10], 'penalty': ['l1', 'l2']},
              'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100]},
              'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
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
    with open('model_comparison.csv', 'w') as csv_file:
      table = csv.writer(csv_file, delimiter = ',')
      table.writerow(['ID', 'MODEL', 'PARAMETERS', 'ACCURACY', 'PRECISION', 'RECALL', 'AUC', 'F1'])

      count = 0
      for index, CLASSIFIERS in enumerate([CLASSIFIERS[x] for x in MODELS]):
          values = PARAMETERS[MODELS[index]]
          for param in ParameterGrid(values):
              CLASSIFIERS.set_params(**param)
              CLASSIFIERS.fit(x_train, y_train)
              pred_probs = CLASSIFIERS.predict(x_validate)       
              eval_metrics = evaluate(y_validate, pred_probs)

              table_row = [count, MODELS[index], param]
              for metric in eval_metrics:
                  table_row.append(metric)

              table.writerow(table_row) 
              count += 1 
              precision_recall(y_validate, pred_probs, MODELS[index], count)
   

def evaluate(y_validate, pred_probs):
    '''
    Evaluates the fit of the model on the validation data 
    using various metric.
    '''
    
    ACCURACY = metrics.accuracy_score(y_validate, pred_probs)
    PRECISION = metrics.precision_score(y_validate, pred_probs)
    RECALL = metrics.recall_score(y_validate, pred_probs)
    AUC = metrics.roc_auc_score(y_validate, pred_probs)
    F1 = metrics.f1_score(y_validate, pred_probs)

    return ACCURACY, PRECISION, RECALL, AUC, F1

def precision_recall(y_test, pred_probs, model_name, count):   
    '''
    Plots the precision and recall curves for the various classifiers
    after the models have been fitted.
    '''

    y_score = pred_probs
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    
    name = model_name + str(count)
    plt.title(name)
    plt.savefig('PR_Curves/' + name)

def best_model(eval_metric):
    '''
    Find the best model and associated parameters based on the
    evaluation metric that the user is interested in.
    '''

    model_comp = pd.DataFrame.from_csv('model_comparison.csv')
    
    highest_value = model_comp[eval_metric].max()
    best_model = model_comp[model_comp[eval_metric] == highest_value]
    
    model = best_model.iloc[0][0]
    parameters = best_model.iloc[0][1]
    accuracy = best_model.iloc[0][2]
    precision = best_model.iloc[0][3]
    recall = best_model.iloc[0][4]
    AUC = best_model.iloc[0][5]
    F1 = best_model.iloc[0][6]

    return model, parameters, accuracy, precision, recall, AUC, F1
    
def score_data(best_model, best_parameters, df_test, dep_vars, indep_vars, df_train):
    '''
    Utilizes the best model to predict outcomes in the 
    testing dataset
    '''
    y_train = df_train[dep_vars]
    y_train = np.ravel(y_train)
    x_train = df_train[indep_vars]
    x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = .2, random_state = 0)
    
    model = CLASSIFIERS[best_model]
    x_test = df_test[indep_vars]
    model.fit(x_train, y_train)

    y_test = model.predict_proba(x_test)

    prob_list = []
    for prob in y_test:
        prob_list.append(prob[1])

    df_test[dep_vars] = prob_list
    return(df_test[dep_vars])