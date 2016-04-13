import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
import json
import numpy as np
import pylab as pl


def read_data(datafile):

    return(pd.DataFrame.from_csv(datafile))

def explore_data(df):
    values = df.columns
    
    descriptive = df.describe()
    mode = df.mode()
    missing_values = (len(df.index)) - df.count()

    descriptive.to_csv('descriptive_statistics.csv') 
    mode.to_csv('modes.csv')
    missing_values.to_csv('missing_values.csv')
  
def find_null_values(df):
    df_variables = pd.melt(df)
    null_data = df_variables.value.isnull()
    null_df = pd.crosstab(df_variables.variable, null_data)
    return null_df

def histogram(df):

    values = df.columns
    for value in values:
        df.hist(value)
        plt.savefig(value)