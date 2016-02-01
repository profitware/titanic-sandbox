# -*- coding: utf-8 -*-

__author__ = 'Sergey Sobko'

from wine import Wine


if __name__ == '__main__':
    wine = Wine('wine.data')

    wine.choose_best_classifier()

    with open('task3_1.txt', 'w') as f:
        f.write(wine.out_k_original)

    with open('task3_2.txt', 'w') as f:
        f.write(wine.out_max_accuracy_original)

    with open('task3_3.txt', 'w') as f:
        f.write(wine.out_k_scaled)

    with open('task3_4.txt', 'w') as f:
        f.write(wine.out_max_accuracy_scaled)
