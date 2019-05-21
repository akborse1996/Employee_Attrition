# -*- coding: utf-8 -*-
"""
Created on Fri May 10 11:49:09 2019

@author: amol.borse
"""

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

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 20))
classifier.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,

beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
moving_variance_initializer='ones'))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))


# Compiling the ANN
classifier.compile(optimizer = 'RMSprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 20)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
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
