# -*- coding: utf-8 -*-

__author__ = 'Sergey Sobko'

from titanic import Titanic


if __name__ == '__main__':
    titanic = Titanic('titanic.csv')

    with open('task2.txt', 'w') as f:
        f.write(titanic.survival_criteria)
