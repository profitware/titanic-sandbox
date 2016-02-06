#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Sergey Sobko'
__email__ = 'S.Sobko@profitware.ru'
__copyright__ = 'Copyright 2016, The Profitware Group'


from numpy import linspace
from sklearn.cross_validation import (
    KFold,
    cross_val_score
)
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale


class Boston(object):
    boston_data = None

    p_accurate = -1
    max_accuracy = None

    def __init__(self):
        self.boston_data = load_boston()
        
    def choose_best_regressor(self):
        boston_len = len(self.boston_data.target)
    
        classes = self.boston_data.target
    
        criteria = self.boston_data.data
        scaled_criteria = scale(criteria)
    
        cv_gen = KFold(boston_len, n_folds=5, shuffle=True, random_state=42)

        for p in linspace(1, 10, 200):
            regressor = KNeighborsRegressor(
                n_neighbors=5,
                weights='distance',
                p=p,
                metric='minkowski'
            )
    
            get_accuracy = lambda criteria_data: cross_val_score(
                    estimator=regressor,
                    X=criteria_data,
                    y=classes,
                    scoring='mean_squared_error',
                    cv=cv_gen,
                    verbose=True
            )

            accuracy = get_accuracy(scaled_criteria)

            accuracy_mean = accuracy.mean()

            if self.max_accuracy is None or accuracy_mean > self.max_accuracy:
                self.p_accurate, self.max_accuracy = p, accuracy_mean

    @property
    def out_p_accurate(self):
        return '{0:.2f}'.format(self.p_accurate)

    @property
    def out_max_accuracy(self):
        return '{0:.2f}'.format(self.max_accuracy)
