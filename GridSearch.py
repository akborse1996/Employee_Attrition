
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 12:44:24 2019

@author: amol.borse
"""
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('D:/train.csv')
X = dataset.iloc[:, 0:9]
y = dataset.iloc[:, -1:]
a=['Department','salary']
X = pd.get_dummies(X, columns=a)
y=y.values

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Grid search to find optimal parameters
from sklearn.model_selection import GridSearchCV

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.01],
                     'C': [10,100]},
                    {'kernel': ['linear'], 'gamma': [0.01],'C': [10,100]}]


from sklearn.svm import SVC
clf = GridSearchCV(SVC(C=1), tuned_parameters, n_jobs=4, cv=2,
                       scoring='f1_macro')
clf = clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)


print(clf.best_params_)
print(clf.best_score_)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score,precision_score,recall_score
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

