# -*- coding: utf-8 -*-

__author__ = 'Sergey Sobko'

from perceptron import MyPerceptron


if __name__ == '__main__':
    perceptron = MyPerceptron('perceptron-train.csv', 'perceptron-test.csv')

    perceptron.get_accuracy_diff()

    with open('task5.txt', 'w') as f:
        f.write(perceptron.out_accuracy_diff)
