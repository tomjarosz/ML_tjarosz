import pandas as pd
import json
import numpy as np

def missing_values_means(df, variable, impute_value):
    '''
    Fills NA values in a dataframe with the impute value if it 
    is provided, otherwise fills NA value with the variable mean.
    '''
    if impute_value:
        df[variable].fillna(impute_value, inplace = True)

    else:
        df[variable].fillna(df[variable].mean(), inplace = True)
        return df, df[variable].mean()
 
def missing_values_zero(df, variable):
    '''
    Fills NA values in a dataframe with 0
    '''
    df[variable].fillna(0, inplace = True)
    return df

def missing_values_cond_mean(df, variable, cond_on):
    '''
    Fills NA values in a dataframe with conditional means 
    that are determined by the user and specific dataset
    '''
    # customize the conditional mean function based on the dataset
    pass

def discretize_continuous_var (df, bins, var_labels, depend_var, column_name):
    '''
    Creates and groups continuous variables into user-defined
    categories
    '''
    categ_var = pd.cut(df[depend_var], bins, labels = var_labels)
    df[column_name] = categ_var

def categ_to_binary(df, var_labels, depend_var, column_name): 
    '''
    Creates and groups categorical variables into 
    respective binary variable bins
    '''
    for bin in var_labels:
        df[depend_var + bin] = 0
        df.loc[(df[column_name] == bin), (depend_var + bin)] = 1