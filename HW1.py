import pandas as pd
import urllib.request
import json

data = pd.read_csv('mock_student_data.csv')
df = pd.DataFrame(data, )


df.hist()
plt.savefig('histogram')

#Summary Statisitics:
print("Summary statistics")
print(df.describe())
print(df.mode())

# Missing data:
print("Missing Data")
print((len(df.index)) - df.count())

def genderize(df):

    no_gender = df[df['Gender'].isnull()]['First_name']
    for name in no_gender:
        gender = gender_api(name)

        df.loc[df['Gender'].isnull(),'Gender'] = gender

    return df
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
    age_mean = round(graduated['Age'].mean(), 1)
    GPA_mean = round(graduated['GPA'].mean(), 1)
    days_missed_mean = round(graduated['Days_missed'].mean(), 1)

    df['Age'].fillna(age_mean, inplace = True)
    df['GPA'].fillna(GPA_mean, inplace = True)
    df['Days_missed'].fillna(days_missed_mean, inplace = True)

    df.to_csv('missing_values_cond_means.csv')