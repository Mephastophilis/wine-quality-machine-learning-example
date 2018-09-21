# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV
import sklearn.metrics
import numpy as np

reds = pd.read_csv('winequality-red.csv', sep=';')
X = reds.values[:, :11]
y = reds.values[:, 11]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=30, test_size=0.15)

lr = LinearRegression(normalize=True)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print('Mean aboslute error for linear regression = ' + str(sklearn.metrics.mean_absolute_error(y_test, y_pred_lr)))
print('Mean squared error for linear regression = ' + str(sklearn.metrics.mean_squared_error(y_test, y_pred_lr)))
print('R^2 score for linear regression = ' + str(sklearn.metrics.r2_score(y_test, y_pred_lr)) + '\n')

alphas_list = np.arange(0.01,10,0.01)
reg = RidgeCV(alphas=alphas_list, cv=None, fit_intercept=True, scoring=None, normalize=False)
reg.fit(X_train, y_train)       
print('Alpha for regularized linear regression = ' + str(reg.alpha_))
y_pred_reg = reg.predict(X_test)
print('Mean aboslute error for regularized linear regression = ' + str(sklearn.metrics.mean_absolute_error(y_test, y_pred_reg)))
print('Mean squared error for regularized linear regression = ' + str(sklearn.metrics.mean_squared_error(y_test, y_pred_reg)))
print('R^2 score for regularized linear regression = ' + str(sklearn.metrics.r2_score(y_test, y_pred_reg)) + '\n')


