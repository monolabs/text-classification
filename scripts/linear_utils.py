import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge, Lasso, lasso_path, LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import LinearSVC, SVC

from typing import Callable

import matplotlib.pyplot as plt
import seaborn as sns




def train_logit_cv(X:np.ndarray, y: np.ndarray, metric:Callable, k:int=5, alpha:float=0.0, random_state:int=None)->float:
    '''
    trains logistic regression and return score.
    Args:
        X: m x d array
        y: m x 1 array
        alpha: regularization parameter
        random_state: sed to be used for k-fold splitting and optimization
    Returns:
        float: metric
    '''
    kf = KFold(n_splits=k, random_state=random_state, shuffle=True)
    fold_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        model = LogisticRegression(max_iter=10000, class_weight='balanced', C=alpha, random_state=random_state)
        model.fit(X_train, y_train)
    
        y_hat = model.predict(X_test)
        score = metric(y_test, y_hat)
        fold_scores.append(score)
    
    return np.array(fold_scores).mean()



def train_svm_cv(X:np.ndarray, y: np.ndarray, metric:Callable, k:int=5, alpha:float=0.0, random_state:int=None)->float:
    '''
    trains logistic regression and return score.
    Args:
        X: m x d array
        y: m x 1 array
        alpha: regularization parameter
        random_state: sed to be used for k-fold splitting and optimization
    Returns:
        float: metric
    '''
    kf = KFold(n_splits=k, random_state=random_state, shuffle=True)
    fold_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        model = LinearSVC(max_iter=100000, class_weight='balanced', C=alpha, random_state=random_state)
        model.fit(X_train, y_train)
    
        y_hat = model.predict(X_test)
        score = metric(y_test, y_hat)
        fold_scores.append(score)
    
    return np.array(fold_scores).mean()