# -*- coding: utf-8 -*-

__author__ = 'Sergey Sobko'

from titanic import Titanic


if __name__ == '__main__':
    titanic = Titanic('titanic.csv')

    with open('task1_1.txt', 'w') as f:
        f.write(titanic.titanic_sex)

    with open('task1_2.txt', 'w') as f:
        f.write(titanic.survived_percentage)

    with open('task1_3.txt', 'w') as f:
        f.write(titanic.first_class_percentage)

    with open('task1_4.txt', 'w') as f:
        f.write(titanic.std_and_mean_for_age)

    with open('task1_5.txt', 'w') as f:
        f.write(titanic.correlation_sibsp_parch)

    with open('task1_6.txt', 'w') as f:
        f.write(titanic.most_popular_female_name)
