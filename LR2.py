# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:51:24 2019

@author: kylesun
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline

pipe = make_pipeline(
    SMOTE(),
    LogisticRegression()
)

# Fit..
pipe.fit(X_train, y_train)

# Predict..
y_pred = pipe.predict(X_test)

from sklearn import metrics

# Evaluate the model
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))

import warnings
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore", category=DeprecationWarning) 

pipe = make_pipeline(
    SMOTE(),
    LogisticRegression()
)

weights = np.linspace(0.005, 0.05, 10)

gsc = GridSearchCV(
    estimator=pipe,
    param_grid={
        #'smote__ratio': [{0: int(num_neg), 1: int(num_neg * w) } for w in weights]
        'smote__ratio': weights
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

pipe = make_pipeline(
    SMOTE(ratio=0.015),
    LogisticRegression()
)

# Fit..
pipe.fit(X_train, y_train)

# Predict..
y_pred = pipe.predict(X_test)

# Evaluate the model
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))