#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Sergey Sobko'
__email__ = 'S.Sobko@profitware.ru'
__copyright__ = 'Copyright 2016, The Profitware Group'


from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler


class MyPerceptron(object):
    perceptron_train_data = None
    perceptron_test_data = None

    accuracy_diff = None

    def __init__(self, train_csv, test_csv):
        self.perceptron_train_data = read_csv(train_csv)
        self.perceptron_test_data = read_csv(test_csv)

    def get_accuracy_diff(self):
        scaler = StandardScaler()

        def count_accuracy(train_data, train_target, true_data, true_target):
            clf = Perceptron(random_state=241)
            clf.fit(train_data, train_target)
            pred_target = clf.predict(true_data)

            return accuracy_score(true_target, pred_target)

        train_target = self.perceptron_train_data.ix[:,0].ravel()
        train_data = self.perceptron_train_data.ix[:,1:]

        test_target = self.perceptron_test_data.ix[:,0].ravel()
        test_data = self.perceptron_test_data.ix[:,1:]

        non_scaled_accuracy = count_accuracy(train_data, train_target, test_data, test_target)

        scaled_accuracy = count_accuracy(
            scaler.fit_transform(train_data),
            train_target,
            scaler.transform(test_data),
            test_target
        )

        self.accuracy_diff = scaled_accuracy - non_scaled_accuracy

    @property
    def out_accuracy_diff(self):
        return '{0:.3f}'.format(self.accuracy_diff)
