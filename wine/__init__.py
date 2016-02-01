#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Sergey Sobko'
__email__ = 'S.Sobko@profitware.ru'
__copyright__ = 'Copyright 2016, The Profitware Group'


from pandas import read_csv
from sklearn.cross_validation import (
    KFold,
    cross_val_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale


class Wine(object):
    wine_data = None

    k_original = -1
    max_accuracy_original = 0

    k_scaled = -1
    max_accuracy_scaled = 0

    def __init__(self, wine_csv):
        self.wine_data = read_csv(wine_csv)
        
    def choose_best_classifier(self):
        wine_len = len(self.wine_data)
    
        classes = self.wine_data.ix[:,0].ravel()
    
        criteria = self.wine_data.ix[:,1:]
        scaled_criteria = scale(criteria)
    
        cv_gen = KFold(wine_len, n_folds=5, shuffle=True, random_state=42)

        for k in range(1, 51):
            classifier = KNeighborsClassifier(n_neighbors=k)
    
            get_accuracy = lambda criteria_data: cross_val_score(
                    estimator=classifier,
                    X=criteria_data,
                    y=classes,
                    scoring='accuracy',
                    cv=cv_gen,
                    verbose=True
            ).mean()
    
            accuracy_original = get_accuracy(criteria)
            if accuracy_original > self.max_accuracy_original:
                self.k_original, self.max_accuracy_original = k, accuracy_original
    
            accuracy_scaled = get_accuracy(scaled_criteria)
            if accuracy_scaled > self.max_accuracy_scaled:
                self.k_scaled, self.max_accuracy_scaled = k, accuracy_scaled

    @property
    def out_k_original(self):
        return '{0}'.format(self.k_original)

    @property
    def out_max_accuracy_original(self):
        return '{0:.2f}'.format(self.max_accuracy_original)

    @property
    def out_k_scaled(self):
        return '{0}'.format(self.k_scaled)

    @property
    def out_max_accuracy_scaled(self):
        return '{0:.2f}'.format(self.max_accuracy_scaled)
