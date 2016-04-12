import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
import json
import numpy as np
import pylab as pl
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics

INDEP_VARS = ['RevolvingUtilizationOfUnsecuredLines', #'age'#,
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 
       #'MonthlyIncome'#,
       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
       'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
       'NumberOfDependents', #'AgeBins'#, 
       'AGE<20', 'AGE20-30', 'AGE30-40',
       'AGE40-50', 'AGE50-60', 'AGE60-70', 'AGE70-80', 'AGE80-90', 'AGE90-100',
       'AGE>100', #'MonthlyIncomeBins'#, 
       'INCOME0-5000', 'INCOME5000-10000',
       'INCOME10000-15000', 'INCOME15000-20000', 'INCOME20000-25000',
       'INCOME25000-30000', 'INCOME30000-35000', 'INCOME35000-40000',
       'INCOME>40000']
DEP_VARS = ['SeriousDlqin2yrs']

CONTIN_VAR = ['MonthlyIncome', 'age']
CATEG_VAR = []

train_data = pd.DataFrame.from_csv('cs-training.csv')
test_data = pd.DataFrame.from_csv('cs-test.csv')

def read_data(datafile):
    dataframe = pd.DataFrame.from_csv(datafile)
    return dataframe

def explore_data(df):

    values = df.columns
    data = {}
    desc_stats = df.describe() 
    for value in values:
        info = []
        mean = desc_stats[value]['mean']
        median = desc_stats[value]['50%']
        SD = desc_stats[value]['std']
        missing_data = (len(df.index)) - df.count()[value]
        mode = df.mode()[value]

        # possible_modes = []
        # for x in range(len(mode)):
        #     possible_modes.append(mode[x])
        
        # modes=[]
        # for x in possible_modes:
        #     if x > 0 or x < 0:
        #         modes.append(x)
    
        info.extend([mean, median, modes, SD, missing_data])
        data[value] = info

    rows = ['Mean', 'Median', 'Mode(s)', 'Standard Deviation', 'Missing Values']

    stats = pd.DataFrame(data, rows)
    stats.to_csv('descriptive_statistics.csv')   

def histogram(df):
    
    df.hist()
  
def find_null_values(df):
    df_variables = pd.melt(df)
    null_data = df_variables.value.isnull()
    null_df = pd.crosstab(df_variables.variable, null_data)
    return null_df

def missing_values_means(df, variable):
    
    df[variable].fillna(df[variable].mean(), inplace = True)
    return df
 
def missing_values_zero(df, variable):

    df[variable].fillna(0, inplace = True)
    return df

def missing_values_cond_mean(df, variable, cond_on):
    #customize the conditional mean function based on the dataset
    pass

def categ_to_binary (df, CATEG_VAR):
    
    for variable in CATEG_VAR:
        df_categ = pd.get_dummies(df[CATEG_VAR])
        df_new = df.join(df_categ)

    return df_new     

def discretize_continuous_var (df):

    age_bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150]
    age_groups = ['<20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90-100','>100']
    age = pd.cut(df['age'], age_bins, labels = age_groups)
    df['AgeBins'] = age

    for bins in age_groups:
        df['AGE' + bins] = 0
        df.loc[(df['AgeBins'] == bins), ('AGE' + bins)] = 1
    
    monthly_income_bins = [-1, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 10000000000000]
    monthly_income_groups = ['0-5000','5000-10000','10000-15000','15000-20000','20000-25000','25000-30000','30000-35000','35000-40000','>40000']
    monthly_income = pd.cut(df['MonthlyIncome'], monthly_income_bins, labels = monthly_income_groups)
    df['MonthlyIncomeBins'] = monthly_income

    for bins in monthly_income_groups:
          df['INCOME' + bins] = 0
          df.loc[(df['MonthlyIncomeBins'] == bins), ('INCOME' + bins)] = 1

    return df
    

def logistic_regression(df_train, df_test):

    y_train = df_train[DEP_VARS]
    y_train = np.ravel(y_train)
    x_train = df_train[INDEP_VARS]

    y_test = df_test[DEP_VARS]
    x_test = df_test[INDEP_VARS]

    logit = LogisticRegression() 
    logit = logit.fit(x_train, y_train)
    score = logit.score(x_train, y_train)

    y_test = logit.predict(x_test)
    y_test_prob = logit.predict_proba(x_test)

    df_test[DEP_VARS] = y_test
    print(y_test_prob)

def prepare_data(df_train):

    y_train = df_train[DEP_VARS]
    y_train = np.ravel(y_train)
    x_train = df_train[INDEP_VARS]

    x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = .2, random_state = 0)
   

def logit2(x_train, x_validate, y_train, y_validate):

    logit2 = LogisticRegression()
    logit2.fit(x_train, y_train)

    # predict_y = logit2.predict(x_validate)
    probs = logit2.predict_proba(x_validate)

#eval method
    accuracy = metrics.accuracy_score(y_validate, probs)
    print(accuracy)

def score_data(model, df_test):
    
    x_test = df_test[INDEP_VARS]    
    y_test = df_test[DEP_VARS]

    #logit2.predict
   y_test =  model.predict_proba(x_test)
   print y_test[0]

    take the [0, 1]-- want the prob of 1
#     confusion_matrix = metrics.confusion_matrix(y_test, predicted)
#     precision_score = metrics.precision_score(y_test, predicted)
#     recall_score = metrics.recall_score(y_test, predicted)
#     print(accuracy)

missing_values_means(train_data, 'MonthlyIncome')
missing_values_zero(train_data, 'NumberOfDependents')
missing_values_means(test_data, 'MonthlyIncome')
missing_values_zero(test_data, 'NumberOfDependents')


discretize_continuous_var(train_data)
discretize_continuous_var(test_data)   

# logistic2_regression(train_data)
# print(train_data)
# logistic_regression(train_data, test_data)