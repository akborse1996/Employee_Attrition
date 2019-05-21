# k-Fold Cross Validation

import pandas as pd
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
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf',gamma='scale', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("mean score provided by k-fold:")
print(accuracies.mean())
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
