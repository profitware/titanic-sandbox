# -*- coding: utf-8 -*-

__author__ = 'Sergey Sobko'

from boston import Boston


if __name__ == '__main__':
    boston = Boston()

    boston.choose_best_regressor()

    with open('task4.txt', 'w') as f:
        f.write(boston.out_p_accurate)
