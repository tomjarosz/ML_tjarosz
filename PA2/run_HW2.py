from read_explore import *
from process_data import *
from models import *

INDEP_VARS = ['RevolvingUtilizationOfUnsecuredLines', 'age',
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
       'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
       'NumberOfDependents', 'AGE<20',
       'AGE20-30', 'AGE30-40', 'AGE40-50', 'AGE50-60', 'AGE60-70', 'AGE70-80',
       'AGE80-90', 'AGE90-100', 'AGE>100', 'MONTHLYINCOME0-5000',
       'MONTHLYINCOME5000-10000', 'MONTHLYINCOME10000-15000',
       'MONTHLYINCOME15000-20000', 'MONTHLYINCOME20000-25000',
       'MONTHLYINCOME25000-30000', 'MONTHLYINCOME30000-35000',
       'MONTHLYINCOME35000-40000', 'MONTHLYINCOME>40000']

DEP_VARS = ['SeriousDlqin2yrs']

CONTIN_VAR = ['MonthlyIncome', 'age']
CATEG_VAR = []

test_data = read_data('cs-test.csv')
train_data = read_data('cs-training.csv')

explore_data(train_data)
histogram(train_data)

missing_values_means(train_data, 'MonthlyIncome')
missing_values_zero(train_data, 'NumberOfDependents')

missing_values_means(test_data, 'MonthlyIncome')
missing_values_zero(test_data, 'NumberOfDependents')

age_bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150]
age_labels = ['<20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90-100','>100']
discretize_continuous_var(train_data, age_bins, age_labels, 'age', 'AgeBins')
discretize_continuous_var(test_data, age_bins, age_labels, 'age', 'AgeBins')

monthly_income_bins = [-1, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 10000000000000]
monthly_income_labels = ['0-5000','5000-10000','10000-15000','15000-20000','20000-25000','25000-30000','30000-35000','35000-40000','>40000']
discretize_continuous_var(train_data, monthly_income_bins, monthly_income_labels, 'MonthlyIncome', 'MonthlyIncomeBins')
discretize_continuous_var(test_data, monthly_income_bins, monthly_income_labels, 'MonthlyIncome', 'MonthlyIncomeBins')
#add labels to INDEP_VARS

categ_to_binary(train_data, age_labels, 'AGE', 'AgeBins')
categ_to_binary(test_data, age_labels, 'AGE', 'AgeBins')

categ_to_binary(train_data, monthly_income_labels, 'MONTHLYINCOME', 'MonthlyIncomeBins')
categ_to_binary(test_data, monthly_income_labels, 'MONTHLYINCOME', 'MonthlyIncomeBins')


x_train, x_validate, y_train, y_validate = prep_model_data(train_data, DEP_VARS, INDEP_VARS)
pred_probs, logit_model = logit_model(x_train, x_validate, y_train, y_validate)
accuracy = evaluate(y_validate, pred_probs)

result = score_data(logit_model, test_data, DEP_VARS, INDEP_VARS)
result.to_csv('deliquency_prediction.csv')