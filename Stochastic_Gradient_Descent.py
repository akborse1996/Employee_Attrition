# -*- coding: utf-8 -*-
"""
Created on Wed May  8 18:09:47 2019

@author: amol.borse
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:34:09 2019

@author: amol.borse
"""
import tensorflow 
from tensorflow import keras
import numpy as np

import pandas as pd

# Importing the dataset
dataset = pd.read_csv('D:/train.csv')
X = dataset.iloc[:, 0:9]
y = dataset.iloc[:, -1:]
a=['Department','salary']
X = pd.get_dummies(X, columns=a)
X=X.values
y=y.values

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 20))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
from keras import optimizers

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# Compiling the ANN
classifier.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 20)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.metrics import f1_score,precision_score,recall_score
precision=precision_score(y_test,y_pred)
print("precision:") 
print(precision)
recall=recall_score(y_test,y_pred)
print("recall_score:") 
print(recall)
f1_score= f1_score(y_test,y_pred)
print(" f1_score:") 
print( f1_score)

