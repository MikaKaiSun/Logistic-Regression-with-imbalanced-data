# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 18:57:12 2019

@author: kylesun
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("D:\\lottery199\\231.csv")

# Separata data into X/y
y = data['lottery'].values
X = data.drop(['iuin', 'lottery'], axis=1).values

num_neg = (y==0).sum()
num_pos = (y==1).sum()

# Scaling..
scaler = RobustScaler()
X = scaler.fit_transform(X)

# Split into train/test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

import seaborn as sns

print(data.groupby('lottery').size())

sns.countplot(x="lottery", data=data)

from sklearn.linear_model import LogisticRegression
#plain lr it is pretty bad!!

'''
from sklearn import metrics

lr = LogisticRegression()

# Fit..
lr.fit(X_train, y_train)

# Predict..
y_pred = lr.predict(X_test)

# Evaluate the model
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))


#still bad!!!
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix
from matplotlib import pyplot as plt
lr = LogisticRegression(class_weight='balanced')

# Fit..
lr.fit(X_train, y_train)

# Predict..
y_pred = lr.predict(X_test)

# Evaluate the model
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
'''
from sklearn.model_selection import GridSearchCV

weights = np.linspace(0.05, 0.95, 20)

gsc = GridSearchCV(
    estimator=LogisticRegression(),
    param_grid={
        'class_weight': [{0: x, 1: 1.0-x} for x in weights]
    },
    scoring='f1',
    cv=3
)
grid_result = gsc.fit(X, y)

print("Best parameters : %s" % grid_result.best_params_)

# Plot the weights vs f1 score
dataz = pd.DataFrame({ 'score': grid_result.cv_results_['mean_test_score'],
                       'weight': weights })
dataz.plot(x='weight')


lr = LogisticRegression(**grid_result.best_params_)

# Fit..
lr.fit(X_train, y_train)

# Predict..
y_pred = lr.predict(X_test)

from sklearn import metrics

# Evaluate the model
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
