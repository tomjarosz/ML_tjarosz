The goal of this assignment is to build a simple, modular, extensible, machine learning pipeline in Python. The pipeline should have functions that can do the following tasks:

1. Read Data: For this assignment, assume input is CSV
2. Explore Data: You can use the code you wrote for assignment 1 here to generate distributions and data summaries
3. Pre-Process Data: Fill in misssing values
4. Generate Features: Write a sample function that can discretize a continuous variable and one function that can take a categorical variable and create binary variables from it.

panda.getdummies()
5. Build Classifier: For this assignment, select any classifer you feel comfortable with (Logistic Regression for example)

THI IS THE NONGENERALIZED PART: bins: create a bin dictionary with key as the column name and value as three-part tuple with name of first bin, lower bound first bin, upper bound first bin not unclusive, .........
---then use pd.cut to make the bins

6. Evaluate Classifier: you can use any metric you choose for this assignment (accuracy is the easiest one)

The data can be downloaded from https://www.kaggle.com/c/GiveMeSomeCredit/data

There is useful code in ipython notebooks in this git repo: https://github.com/yhat/DataGotham2013/

*fill out probabaility of the deliquent in the test data in the first column



read files 5 different ways. so if csv, use this or that

plot-if conitnuous variable and plotting- (takes min and max and generate bins)- can hard code this for plotting

4 ways to fill missing data- if gender use this, if age use that, etc

modle- takes all the 1. x (list of column names) and 2. the target (list y column names) 3. df-- always takes these into the function whenever. 
    -then when take model 1 of 30, take the df, x columns, and y columns
    -scikit learn is the metohds for the different mofles we can run 
    -__.fit(___, __) is the general function to fit all the data

columns names = 
y variable = 
drop these columns = 
transform these yes and no answers to 1 and 0 

train on 90% and test on 10% of training data bc we know the answer
test on data set to fill in 
check accuracy of the model based on results of train/test

function(train).test()    