import pandas as pd
import urllib.request
import json

data = pd.read_csv('mock_student_data.csv')
df = pd.DataFrame(data, )


# df.hist()
# plt.savefig('histogram')

# #Summary Statisitics:
# print("Summary statistics")
# print(df.describe())
# print(df.mode())

# # Missing data:
# print("Missing Data")
# print((len(df.index)) - df.count())

def genderize(df):

    no_gender = df[df['Gender'].isnull()]['First_name']
    for name in no_gender:
        gender = gender_api(name)

        df.loc[df['Gender'].isnull(),'Gender'] = gender

    df.to_csv('genders_filled_in')    
        

def gender_api(first_name):

    webservice_url = "https://api.genderize.io/?name=" + first_name
    gender_data = json.loads(urllib.request.urlopen(webservice_url).read().decode("utf8"))
    return gender_data["gender"]


print(genderize(df))    


