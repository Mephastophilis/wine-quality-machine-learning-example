# -*- coding: utf-8 -*-
"""
wine quality regression and classification

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV
import sklearn.metrics
from sklearn.svm import SVC
#from sklearn.cross_validation import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('winequality-red.csv', sep=';')
X = df.values[:, :11]
y = df.values[:, 11]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=30, test_size=0.15)

lr = LinearRegression(normalize=True)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print('Mean aboslute error for linear regression = ' + str(sklearn.metrics.mean_absolute_error(y_test, y_pred_lr)))
print('Mean squared error for linear regression = ' + str(sklearn.metrics.mean_squared_error(y_test, y_pred_lr)))
print('R^2 score for linear regression = ' + str(sklearn.metrics.r2_score(y_test, y_pred_lr)) + '\n')

alphas_list = np.arange(0.001,10,0.001)
reg = RidgeCV(alphas=alphas_list, cv=None, fit_intercept=True, scoring=None, normalize=True)
reg.fit(X_train, y_train)       
print('Alpha for regularized linear regression = ' + str(reg.alpha_))
y_pred_reg = reg.predict(X_test)
print('Mean aboslute error for regularized linear regression = ' + str(sklearn.metrics.mean_absolute_error(y_test, y_pred_reg)))
print('Mean squared error for regularized linear regression = ' + str(sklearn.metrics.mean_squared_error(y_test, y_pred_reg)))
print('R^2 score for regularized linear regression = ' + str(sklearn.metrics.r2_score(y_test, y_pred_reg)) + '\n')

print('SVC fit with linear kernel')
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
print(sklearn.metrics.classification_report(y_test, y_pred_svc))

print('SVC fit with radial basis function kernel')
svc_g = SVC(kernel='rbf')
svc_g.fit(X_train, y_train)
y_pred_svc_g = svc_g.predict(X_test)
print(sklearn.metrics.classification_report(y_test, y_pred_svc_g))

linear_contributions = lr.coef_*np.mean(X, axis=0)
most_relevant_feature_lr = np.where(linear_contributions==max(linear_contributions))[0][0]

fig = plt.scatter(X[:,most_relevant_feature_lr], y)
plt.xlabel(list(df.columns.values)[most_relevant_feature_lr].capitalize())
plt.ylabel(list(df.columns.values)[11].capitalize())
plt.title('Wine Quality vs. ' + str(list(df.columns.values)[most_relevant_feature_lr]).capitalize())