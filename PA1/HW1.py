import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
import json

# data = pd.read_csv('mock_student_data.csv')
# df = pd.DataFrame(data, )

def histogram(df):
    df.loc[:, ['Age', 'GPA', 'Days_missed']].hist()
    plt.savefig('histogram.png')   

def summary_stats(df):
    
    values = ['Age', 'Days_missed', 'GPA']
    data = {}
    desc_stats = df.describe() 
    for value in values:
        info = []
        mean = desc_stats[value]['mean']
        median = desc_stats[value]['50%']
        SD = desc_stats[value]['std']
        missing_data = (len(df.index)) - df.count()[value]
        mode = df.mode()[value]

        possible_modes = []
        for x in range(len(mode)):
            possible_modes.append(mode[x])
        
        modes=[]
        for x in possible_modes:
            if x > 0 or x < 0:
                modes.append(x)
    
        info.extend([mean, median, modes, SD, missing_data])
        data[value] = info

    rows = ['Mean', 'Median', 'Mode(s)', 'Standard Deviation', 'Missing Values']

    stats = pd.DataFrame(data, rows)
    stats.to_csv('descriptive_statistics.csv')   
        
def gender_api(first_name):

    webservice_url = "https://api.genderize.io/?name=" + first_name
    gender_data = json.loads(urllib.request.urlopen(webservice_url).read().decode("utf8"))
    return gender_data["gender"]

def genderize(df):

    no_gender = df[df['Gender'].isnull()]['First_name']
    for name in no_gender:
        gender = gender_api(name)
        df.loc[(df['Gender'].isnull()) & (df['First_name'] == name),'Gender'] = gender

    df.to_csv('genders_filled_in.csv')

def missing_values_means (df):
    df2 = df.copy(deep=True)

    age_mean = round((df['Age'].mean()), 1)
    GPA_mean = round((df['GPA'].mean()), 1)
    days_missed_mean = round((df['Days_missed'].mean()), 1)

    df2['Age'].fillna(age_mean, inplace = True)
    df2['GPA'].fillna(GPA_mean, inplace = True)
    df2['Days_missed'].fillna(days_missed_mean, inplace = True)

    df2.to_csv('missing_values_means.csv')

def missing_values_cond_means (df):
    df3 = df.copy(deep=True)

    graduated = df[df['Graduated'] == 'Yes'][['Age', 'GPA', 'Days_missed']]
    age_grad_mean = round(graduated['Age'].mean(), 1)
    GPA_grad_mean = round(graduated['GPA'].mean(), 1)
    days_missed_grad_mean = round(graduated['Days_missed'].mean(), 1)

    df3.loc[(df['Graduated'] == 'Yes') & (df3['Age'].isnull()),'Age'] = age_grad_mean
    df3.loc[(df['Graduated'] == 'Yes') & (df3['GPA'].isnull()),'GPA'] = GPA_grad_mean
    df3.loc[(df['Graduated'] == 'Yes') & (df3['Days_missed'].isnull()),'Days_missed'] = days_missed_grad_mean

    no_graduated = df[df['Graduated'] == 'No'][['Age', 'GPA', 'Days_missed']]
    age_nograd_mean = round(no_graduated['Age'].mean(), 1)
    GPA_nograd_mean = round(no_graduated['GPA'].mean(), 1)
    days_missed_nograd_mean = round(no_graduated['Days_missed'].mean(), 1)

    df3.loc[(df['Graduated'] == 'No') & (df3['Age'].isnull()),'Age'] = age_nograd_mean
    df3.loc[(df['Graduated'] == 'No') & (df3['GPA'].isnull()),'GPA'] = GPA_nograd_mean
    df3.loc[(df['Graduated'] == 'No') & (df3['Days_missed'].isnull()),'Days_missed'] = days_missed_nograd_mean
    
    df3.to_csv('missing_values_cond_means.csv')

def drop_missing_data(df):
    df4 = df.copy(deep=True)

    df4 = df[(df['Age'].notnull()) & (df['GPA'].notnull()) & (df['Days_missed'].notnull())]
    df4.to_csv('values_with_complete_data.csv')

