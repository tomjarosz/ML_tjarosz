### First Assignment
import pandas as pd
import numpy as np

data = pd.read_csv('mock_student_data.csv')
students = pd.DataFrame(data, columns = ['ID','First_name','Last_name','State','Gender','Age','GPA','Days_missed','Graduated'])

print(students['Gender'].describe())