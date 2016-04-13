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

def discretize_continuous_var (df, bins, var_labels, depend_var, column_name):

    categ_var = pd.cut(df[depend_var], bins, labels = var_labels)
    df[column_name] = categ_var

def categ_to_binary(df, var_labels, depend_var, column_name): 
    
    for bin in var_labels:
        df[depend_var + bin] = 0
        df.loc[(df[column_name] == bin), (depend_var + bin)] = 1