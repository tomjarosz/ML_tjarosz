from read_explore import *
from process_data import *
from models import *
# import pandas as pd
# import matplotlib.pyplot as plt
# import urllib.request
# import json
# import numpy as np
# import pylab as pl
# from sklearn.linear_model import LogisticRegression
# from sklearn.cross_validation import train_test_split
# from sklearn import metrics

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


test_data = read_data('cs-test.csv')
train_data = read_data('cs-training.csv')

# explore_data(train_data)
# histogram(train_data)

missing_values_means(train_data, 'MonthlyIncome')
missing_values_zero(train_data, 'NumberOfDependents')

missing_values_means(test_data, 'MonthlyIncome')
missing_values_zero(test_data, 'NumberOfDependents')

# discretize_continuous_var (df):

categ_to_binary(df, CATEG_VAR):





