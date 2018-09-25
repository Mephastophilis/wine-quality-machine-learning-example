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
import matplotlib.gridspec as gridspec

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

    alphas_list = np.arange(0.001,1,0.001)
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
    
    linear_contributions = abs(lr.coef_*(np.max(X, axis=0)-np.min(X, axis=0))*np.mean(X, axis=0))
    most_relevant_feature_lr = np.where(abs(linear_contributions)==max(abs(linear_contributions)))[0][0]
    linear_contributions_2 = np.delete(linear_contributions, np.where(linear_contributions==max(linear_contributions)))
    sec_most_relevant_feature_lr = np.where(abs(linear_contributions_2)==max(abs(linear_contributions_2)))[0][0]+1
    
    figure = plt.figure(figsize=(8, 4),dpi=200)
    gs  = gridspec.GridSpec(1, 2,  wspace=0.4, hspace=0.2)
    a = figure.add_subplot(gs[0, 0:1])
    a.scatter(X[:,most_relevant_feature_lr], y)
    plt.xlabel(list(df.columns.values)[most_relevant_feature_lr].capitalize())
    plt.ylabel(list(df.columns.values)[11].capitalize())
    plt.title(winetype.capitalize() + ' Wine Quality vs. ' + str(list(df.columns.values)[most_relevant_feature_lr]).capitalize())
    m1 = lr.coef_[most_relevant_feature_lr]*np.mean(X, axis=0)[most_relevant_feature_lr]
    b1 = np.median(y)-m1*np.mean(X[y==np.median(y)], axis=0)[most_relevant_feature_lr]
    x1_1 = (min(y)-b1)/m1
    x2_1 = (max(y)-b1)/m1
    x_plot=np.arange(min(x1_1, x2_1), max(x1_1, x2_1), abs(x1_1-x2_1)/100)
    y_plot=x_plot*m1+b1
    a.plot(x_plot,y_plot)
    
    
    a = figure.add_subplot(gs[0, 1:2])
    a.scatter(X[:,sec_most_relevant_feature_lr], y)
    plt.xlabel(list(df.columns.values)[sec_most_relevant_feature_lr].capitalize())
    plt.ylabel(list(df.columns.values)[11].capitalize())
    plt.title(winetype.capitalize() + ' Wine Quality vs. ' + str(list(df.columns.values)[sec_most_relevant_feature_lr]).capitalize())
    m2 = lr.coef_[sec_most_relevant_feature_lr]*np.mean(X, axis=0)[sec_most_relevant_feature_lr]
    b2 = np.median(y)-m2*np.mean(X[y==np.median(y)], axis=0)[sec_most_relevant_feature_lr]
    x1_2 = (min(y)-b2)/m2
    x2_2 = (max(y)-b2)/m2
    x_plot=np.arange(min(x1_2, x2_2), max(x1_2, x2_2), abs(x1_2-x2_2)/100)
    y_plot=x_plot*m2+b2
    a.plot(x_plot,y_plot)
    

    plt.savefig(winetype + '_wine_plot.png')


wine_data_analyzer('winequality-red.csv', 'red')
wine_data_analyzer('winequality-white.csv', 'white')
