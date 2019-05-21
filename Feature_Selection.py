

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Getting  dataset
data=pd.read_csv('D:/train.csv')
X=data.iloc[:,0:9]
y=data['Attrition']

#Encoding of categorical data
dummy_cols = ['Department', 'salary']
X = pd.get_dummies(X, columns=dummy_cols)



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Applying model
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
import sklearn.feature_selection
import matplotlib.pyplot as plt

model = ExtraTreesClassifier()
model2=RandomForestClassifier()
model.fit(X_train_res, y_train_res)
model2.fit(X_train_res, y_train_res)

# Recursive Feature Elimination
from sklearn.feature_selection import RFE

# create the RFE model and select 4 attributes
rfe = RFE(model, 4)
rfe = rfe.fit(X_train_res, y_train_res)
rfe2 = RFE(model, 4)
rfe2 = rfe.fit(X_train_res, y_train_res)

# summarize the selection of the attributes
print("ForExtraTreesClassifier:By RFE" )
print(rfe.support_)
print(rfe.ranking_)
print("RandomForestClassifier by RFE:" )
print(rfe.support_)
print(rfe.ranking_)
print("ForExtraTreesClassifier by FE:" )
print(model.feature_importances_) 
print("RandomForestClassifier by FE:" )
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization
print("ForExtraTreesClassifier graph:" )
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()
print(" For RandomForestClassifier:" )
feat_importances = pd.Series(model2.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()
