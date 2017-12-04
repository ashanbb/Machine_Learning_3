# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 15:40:11 2017

@author: abogollagama
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 22:14:36 2017

@author: abogollagama
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 20:53:57 2017

@author: abogollagama
"""

# Predict 422 - Assignment 4 - Support Vector Machines 

# seed value for random number generators to obtain reproducible results
RANDOM_SEED = 1

# import base packages into the namespace for this program
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import cross_val_score 
from sklearn.pipeline import Pipeline

# cross-validation scoring code adapted from Scikit Learn documentation
from sklearn.metrics import roc_auc_score

# specify the set of classifiers being evaluated

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
names = ["Logistic_Regression", "SVM"]
classifiers = [LogisticRegression(), 
               Pipeline((
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel="linear", C=1, probability=True)),
                ))]
                      
             

from sklearn.metrics import roc_curve

# importing data set
bank = pd.read_csv('bank.csv', sep = ';')  

# examine the shape of original input data
print(bank.shape)

# drop observations with missing data, if any
bank.dropna()

# examine the shape of input data after dropping missing data
print(bank.shape)

# look at the list of column names, note that y is the response
list(bank.columns.values)

# look at the beginning of the DataFrame
bank.head()

bank.info()

# mapping function to convert text no/yes to integer 0/1
convert_to_binary = {'no' : 0, 'yes' : 1}

# define binary variable for having credit in default
default = bank['default'].map(convert_to_binary)

# define binary variable for having a mortgage or housing loan
housing = bank['housing'].map(convert_to_binary)

# define binary variable for having a personal loan
loan = bank['loan'].map(convert_to_binary)

# define response variable to use in the model
response = bank['response'].map(convert_to_binary)

balance = bank['balance']

# analysis on balance 
balance_mean = np.mean(balance)
print("balance mean",balance_mean)

balance_mean = np.mean(balance)
print(balance_mean)

balance_min = min(balance)
print("Minimum Balance", balance_min)

balance_max = max(balance)
print("Maximum Balance", balance_max)


# gather four explanatory variables and response into a numpy array 
# here we use .T to obtain the transpose for the structure we want
model_data = np.array([np.array(default), np.array(housing), np.array(loan), 
    np.array(balance), np.array(response)]).T

print(default, housing, loan, response, balance)

np.random.seed(RANDOM_SEED)
np.random.shuffle(model_data)

# examine the shape of model_data, which we will use in subsequent modeling
print(model_data.shape)

# the rest of the program should set up the modeling methods
# and evaluation within a cross-validation design

#-------------------------------------------------------------------------
# specify the k-fold cross-validation design
from sklearn.model_selection import KFold

# ten-fold cross-validation employed here
N_FOLDS = 10

# set up numpy array for storing results
cv_results = np.zeros((N_FOLDS, len(names)))

kf = KFold(n_splits = N_FOLDS, shuffle=False, random_state = RANDOM_SEED)
# check the splitting process by looking at fold observation counts
index_for_fold = 0  # fold count initialized 
for train_index, test_index in kf.split(model_data):
    print('\nFold index:', index_for_fold,
          '------------------------------------------')
    
    #   note that 0:model_data.shape[1]-1 slices for explanatory variables
#   and model_data.shape[1]-1 is the index for the response variable    
    X_train = model_data[train_index, 0:model_data.shape[1]-1]
    X_test = model_data[test_index, 0:model_data.shape[1]-1]
    y_train = model_data[train_index, model_data.shape[1]-1]
    y_test = model_data[test_index, model_data.shape[1]-1]   
    print('\nShape of input data for this fold:',
          '\nData Set: (Observations, Variables)')
    print('X_train:', X_train.shape)
    print('X_test:',X_test.shape)
    print('y_train:', y_train.shape)
    print('y_test:',y_test.shape)

    index_for_method = 0  # initialize
    for name, clf in zip(names, classifiers):
        print('\nClassifier evaluation for:', name)
        print('  Scikit Learn method:', clf)
        clf.fit(X_train, y_train)  # fit on the train set for this fold
        # evaluate on the test set for this fold
        y_test_predict = clf.predict_proba(X_test)
        fold_method_result = roc_auc_score(y_test, y_test_predict[:,1]) 
        print('Area under ROC curve:', fold_method_result)
        cv_results[index_for_fold, index_for_method] = fold_method_result
        index_for_method += 1
  
    index_for_fold += 1
    
cv_results_df = pd.DataFrame(cv_results)
cv_results_df.columns = names

print('\n----------------------------------------------')
print('Average results from ', N_FOLDS, '-fold cross-validation\n',
      '\nMethod                 Area under ROC Curve', sep = '')     
print(cv_results_df.mean())   

# ROC plot 
fpr, tpr, thresholds = roc_curve(y_test, y_test_predict[:,1])
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0,1], 'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('False Postive Rate')
    plt.ylabel('True Positive Rate')

plot_roc_curve(fpr, tpr)
plt.show()

# Using the model on a hypotehical Data Set

input_data = pd.read_csv('Test_Bank_4.csv')
input_data.set_index('Customer', drop = True, inplace = True)

X = input_data.loc['A':'P','default':'balance']  # training explanatory variables
Y = input_data.loc['A':'P','response'] # training response variable


clf.fit(X,Y)

# set up array for storing training set predictions
results = input_data.loc['A':'P',:]
# add predicted probabilities to the training sample
results['Prob_NO'] = clf.predict_proba(X)[:,0]
results['Prob_YES'] = clf.predict_proba(X)[:,1]
results['Prediction'] = clf.predict(X)
print('\nTraining set predictions from Naive Bayes model\n',results)
print('\n\nOverall training set accuracy:', clf.score(X,Y))

