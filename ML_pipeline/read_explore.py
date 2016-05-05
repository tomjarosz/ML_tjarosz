import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
import json
import numpy as np
import pylab as pl

def read_data(datafile):
    '''
    Reads data from a csv to a DataFrame
    '''
    return(pd.DataFrame.from_csv(datafile))

def explore_data(df):
    '''
    Produces csv files that include descriptive statsitics, 
    mode, and number of missing values for the variables 
    in a DataFrame
    '''
    values = df.columns
    
    descriptive = df.describe()
    mode = df.mode()
    missing_values = (len(df.index)) - df.count()

    # descriptive.to_csv('descriptive_statistics.csv') 
    # mode.to_csv('modes.csv')
    # missing_values.to_csv('missing_values.csv')
  
def find_null_values(df):
    '''
    Returns a count of null values for each variable
    in a DataFrame
    '''

    df_variables = pd.melt(df)
    null_data = df_variables.value.isnull()
    null_df = pd.crosstab(df_variables.variable, null_data)
    return null_df

def histogram(df):
    '''
    Creates a histogram of data for each variable in a 
    DataFrame
    '''
    values = df.columns
    for value in values:
        df.hist(value)
        # plt.savefig(value)