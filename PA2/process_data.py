import pandas as pd
import json
import numpy as np

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
