# -*- coding: utf-8 -*-
"""
wine quality regression and classification

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV
import sklearn.metrics
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.neural_network import MLPClassifier

def wine_data_analyzer(winedata, winetype):

    df = pd.read_csv(winedata, sep=';')
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
    
    plt.figure()
    plt.scatter(X[:,most_relevant_feature_lr], y)
    plt.xlabel(list(df.columns.values)[most_relevant_feature_lr].capitalize())
    plt.ylabel(list(df.columns.values)[11].capitalize())
    plt.title(winetype.capitalize() + ' Wine Quality as a linear function of ' + str(list(df.columns.values)[most_relevant_feature_lr]).capitalize())
    x_plot=np.arange(np.mean(X[y==min(y)], axis=0)[most_relevant_feature_lr]-0.5, np.mean(X[y==max(y)], axis=0)[most_relevant_feature_lr], 0.1)
    y_plot=x_plot*lr.coef_[most_relevant_feature_lr]*np.mean(X, axis=0)[most_relevant_feature_lr]+(np.median(y)-lr.coef_[most_relevant_feature_lr]*np.mean(X, axis=0)[most_relevant_feature_lr]*np.mean(X[y==np.median(y)], axis=0)[most_relevant_feature_lr])
    plt.plot(x_plot,y_plot)
    plt.savefig(winetype + '_wine_plot.png')

wine_data_analyzer('winequality-red.csv', 'red')
wine_data_analyzer('winequality-white.csv', 'white')
