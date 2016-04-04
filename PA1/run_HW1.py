from HW1 import *

data = pd.read_csv('mock_student_data.csv')
df = pd.DataFrame(data, )

histogram(df)
summary_stats(df)
# genderize(df)
missing_values_means (df)
missing_values_cond_means (df)
drop_missing_data(df)