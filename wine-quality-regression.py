# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import accuracy_score

reds = pd.read_csv('winequality-red.csv', sep=';')
X = reds.values[:, :11]
y = reds.values[:, 11]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=98)

lr = LinearRegression(normalize=True)

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

for x in xrange(len(y_pred)):
    y_pred[x] = round(y_pred[x])

print(accuracy_score(y_test, y_pred))


reg = RidgeCV(alphas=[0.1, 1.0, 10.0])
reg.fit(X, y, [0, .1, 1])       
RidgeCV(alphas=[0.1, 1.0, 10.0], cv=None, fit_intercept=True, scoring=None, normalize=False)
reg.alpha_