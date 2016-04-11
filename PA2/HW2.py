import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
import json
import numpy as np
import pylab as pl
from sklearn.linear_model import LogisticRegression

INDEP_VARS = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']
DEP_VARS = ['SeriousDlqin2yrs']

CONTIN_VAR = ['MonthlyIncome', 'age']
CATEG_VAR = []

train_data = pd.DataFrame.from_csv('cs-training.csv')
test_data = pd.DataFrame.from_csv('cs-test.csv')

def read_data(datafile):

    dataframe = pd.DataFrame.from_csv(datafile)
    return dataframe

def explore_data(df):
    return df.describe()     

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
    pass

def categ_to_binary (df, CATEG_VAR):
    
    for variable in CATEG_VAR:
        df_categ = pd.get_dummies(df[CATEG_VAR])
        df_new = df.join(df_categ)

    return df_new     

def discretize_continuous_var (df):

    age_bins = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110]
    age_groups = ['18-25','25-30','30-35','35-40','40-45','45-50','50-55','55-60','60-65','65-70','70-75','75-80','80-85','85-90','90-95','95-100','100-105','105-110']
    age = pd.cut(df['age'], age_bins, labels = age_groups)
    df['AgeBins'] = age

    monthly_income_bins = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 10000000000000]
    monthly_income_groups = ['0-5000','5000-10000','10000-15000','15000-20000','20000-25000','25000-30000','30000-35000','35000-40000','>40000']
    monthly_income = pd.cut(df['MonthlyIncome'], monthly_income_bins, labels = monthly_income_groups)
    df['MonthlyIncomeBins'] = monthly_income

    return df
    

def logistic_regression(df_train, df_test):

    y_train = df_train[DEP_VARS]
    y_train = np.ravel(y_train)
    x_train = df_train[INDEP_VARS]

    y_test = df_test[DEP_VARS]
    x_test = df_test[INDEP_VARS]

    logreg = LogisticRegression() 
    logreg = logreg.fit(x_train, y_train)
    score = logreg.score(x_train, y_train)
    print(score)

    y_test = logreg.predict(x_test)
    y_test_prob = logreg.predict_proba(x_test)

    df_test[DEP_VARS] = y_test
    # df_test['Probability'] = y_test_prob

    

train_data = pd.DataFrame.from_csv('cs-training.csv')
test_data = pd.DataFrame.from_csv('cs-test.csv')

df_train = missing_values_means(train_data, 'MonthlyIncome')
df_train = missing_values_zero(train_data, 'NumberOfDependents')

df_test = missing_values_means(test_data, 'MonthlyIncome')
df_test = missing_values_zero(test_data, 'NumberOfDependents')

logistic_regression(df_train, df_test)










#model file reads in the training file. mean transform and conditional transform
#input the data and create a 'transformed-training.csv' with conditional means
    #conditional on the output variable
#fit the model
#in the test file we dont have target variables listed. we still get the target column but it is empty. we don't have to compare the test against anything becasue we have nothing to test them against. we produce a file with our hard predictions and the kaggle people have the answer key but we dont know what we have wrong bc that would be exposing the labels. 
#we dont have categorical variables so skip that function in the assignment
#Gustav wants a nice evaluation model and what our test precition would be if we submitted to kaggle

#Cross validation: 
#take 10% of the data and hold it out. Use 90% of the data so we save the 10% as validation but this could overfit the model. 
#we could split several times and avaerage the results. 
#look up k fold/ 10 fold validation method. create 10 buckets of data so that 1 bucket is val and 9 are train and use this until all 10 have been the validation  (shuffle the data before train and validate the frist time; shuffle once then make the folds on ID 1-10 as first, ID 11-20 as next)
#or pick a random bucket to be validate and 90% to be validate (random by use randint selection and dont have to shuffle data.)


#ideal way to set the % of train and validate data is to get 100% to train so instead we train with 99% and validate obsv by obsv