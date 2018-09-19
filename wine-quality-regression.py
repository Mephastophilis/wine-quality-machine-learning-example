# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

reds = pd.read_csv('winequality-red.csv', sep=';')
X = reds.values[:, :11]
y = reds.values[:, 11]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33)

X_means = np.mean(X, axis =0)
X_train_means = np.mean(X_train, axis=0)
X_test_means = np.mean(X_test, axis=0)