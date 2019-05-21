
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:34:09 2019

@author: amol.borse
"""


import pandas as pd

# Importing the dataset
dataset = pd.read_csv('D:/ML/train.csv')
X = dataset.iloc[:, 0:9]
y = dataset.iloc[:, -1:]
a=['Department','salary']
X = pd.get_dummies(X, columns=a)
X=X
y=y
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()

model.fit(X_train, y_train)
#Adding the input layer and the first hidden layer 


# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = model.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
precision=precision_score(y_test,y_pred)
print("precision:") 
print(precision)
recall=recall_score(y_test,y_pred)
print("recall_score:") 
print(recall)
f1_score= f1_score(y_test,y_pred)
print(" f1_score:") 
print( f1_score)
print("Accuracy: Tp+Tn/(Tp+Tn+Fp+Fn):")
print(accuracy_score(y_test, y_pred)) 


