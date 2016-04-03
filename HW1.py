import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
import json

data = pd.read_csv('mock_student_data.csv')
df = pd.DataFrame(data, )

def histogram(df):
    df.hist()
    plt.savefig('histogram')

def summary_stats(df):
    
    values = ['Age', 'GPA', 'Days_missed']
    data = {}
    desc_stats = df.describe() 
    for value in values:
        info = []
        mean = desc_stats[value]['mean']
        median = desc_stats[value]['50%']
        SD = desc_stats[value]['std']
        mode = df.mode()[value]
        missing_data = (len(df.index)) - df.count()[value]

        info.extend([mean, median, SD, mode[0], missing_data])
        data[value] = info

    rows = ['Mean', 'Median', 'Mode', 'Standard Deviation', 'Missing Values']

    stats = pd.DataFrame(data, rows)
    stats.to_csv('descriptive_statistics.csv')
    print(stats)

summary_stats(df)

def genderize(df):

    no_gender = df[df['Gender'].isnull()]['First_name']
    for name in no_gender:
        gender = gender_api(name)

        df.loc[df['Gender'].isnull(),'Gender'] = gender

    df.to_csv('genders_filled_in.csv')    
        

def gender_api(first_name):

    webservice_url = "https://api.genderize.io/?name=" + first_name
    gender_data = json.loads(urllib.request.urlopen(webservice_url).read().decode("utf8"))
    return gender_data["gender"]


# print(genderize(df))    

def missing_values_means (df):
    age_mean = round((df['Age'].mean()), 1)
    GPA_mean = round((df['GPA'].mean()), 1)
    days_missed_mean = round((df['Days_missed'].mean()), 1)

    df['Age'].fillna(age_mean, inplace = True)
    df['GPA'].fillna(GPA_mean, inplace = True)
    df['Days_missed'].fillna(days_missed_mean, inplace = True)

    df.to_csv('missing_values_means.csv')    

def missing_values_cond_means (df):

    graduated = df[df['Graduated'] == 'Yes'][['Age', 'GPA', 'Days_missed']]
    age_grad_mean = round(graduated['Age'].mean(), 1)
    GPA_grad_mean = round(graduated['GPA'].mean(), 1)
    days_missed_grad_mean = round(graduated['Days_missed'].mean(), 1)

    df.loc[(df['Graduated'] == 'Yes') & (df['Age'].isnull()),'Age'] = age_grad_mean
    df.loc[(df['Graduated'] == 'Yes') & (df['GPA'].isnull()),'GPA'] = GPA_grad_mean
    df.loc[(df['Graduated'] == 'Yes') & (df['Days_missed'].isnull()),'Days_missed'] = days_missed_grad_mean

    no_graduated = df[df['Graduated'] == 'No'][['Age', 'GPA', 'Days_missed']]
    age_nograd_mean = round(no_graduated['Age'].mean(), 1)
    GPA_nograd_mean = round(no_graduated['GPA'].mean(), 1)
    days_missed_nograd_mean = round(no_graduated['Days_missed'].mean(), 1)

    df.loc[(df['Graduated'] == 'No') & (df['Age'].isnull()),'Age'] = age_nograd_mean
    df.loc[(df['Graduated'] == 'No') & (df['GPA'].isnull()),'GPA'] = GPA_nograd_mean
    df.loc[(df['Graduated'] == 'No') & (df['Days_missed'].isnull()),'Days_missed'] = days_missed_nograd_mean
    
    df.to_csv('missing_values_cond_means.csv')