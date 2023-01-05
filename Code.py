#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 19:10:11 2022

@author: mohamaddalati
"""
###             ANN model
import pandas as pd

# import the data 
df = pd.read_csv('/Users/mohamaddalati/Desktop/INSY-662/Sheet4.csv')

###             Task 1: Pre-Process the data 
##1. Check if there are any missing values
display(df.isnull().any())
# drop observations with one or more missing value 
df.dropna(inplace = True)

##2. Drop irrelevant predictor if exits
# Check if the variable has the same values in all observations so we can drop it 
df['Type'].value_counts()  #since one value is H while the rest is C , I will keep the variable 

##3. Dummify categorical predictors (we only have Manuf and Type variables as categorical variables)
dummydata = pd.get_dummies(df, columns = ['Manuf','Type'])


###             Task 2: Construct predictors and target variables 
X = dummydata.drop(columns=['Name', 'Rating_Binary']) #everything else is predictor 
y = dummydata['Rating_Binary'] 

###             Task3: Standardization using Z-score 
from sklearn.preprocessing import StandardScaler # this is using Z-score Standardization 
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)
scaled_X = pd.DataFrame(scaled_X, columns=X.columns) # transform into a dataframe and add column names
scaled_X.head()

###             Task4: Build an ANN model using MLPClassifier 
from sklearn.neural_network import MLPClassifier #short for Multi-Layer_Perceptor
from sklearn.model_selection import cross_val_score
import numpy as np
# We are going to use GridSearchCV to find the optimal size of hidden layers(i.e. number of nodes) before fitting the model 
from sklearn.model_selection import GridSearchCV
model = MLPClassifier(verbose = True, random_state = 5)

tuning_parameters = { 'hidden_layer_sizes': np. arange(1,22) }

ANN_search = GridSearchCV(model, tuning_parameters, cv = 5, return_train_score = False, n_jobs=1) #

ANN_search.fit(scaled_X, y) #fit the model on the entire dataset (with standardization)
print('Accuracy score is:', ANN_search.best_score_) # to get the accuracy score 
print('Best Parameters:', ANN_search.best_params_) # To find the optimal hidden layer size or number of nodes 






