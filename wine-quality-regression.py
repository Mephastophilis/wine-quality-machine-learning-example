# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

reds = pd.read_csv('winequality-red.csv', sep=';')
X = reds.values[:, :11]
y = reds.values[:, 11]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=98)

lr = LinearRegression(normalize=True)

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

error = sum(abs(y_pred - y_test))

