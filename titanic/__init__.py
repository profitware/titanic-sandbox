# -*- coding: utf-8 -*-

__author__ = 'Sergey Sobko'


from pandas import read_csv


class Titanic(object):
    titanic_data = None

    def __init__(self, titanic_csv):
        self.titanic_data = read_csv(titanic_csv, index_col='PassengerId')

    def _percents(self, field):
        return self.titanic_data.groupby(field).size().apply(
            lambda x: float(x) / self.titanic_data.groupby(field).size().sum() * 100
        )

    @property
    def titanic_sex(self):
        return '{male} {female}'.format(**self.titanic_data['Sex'].value_counts())

    @property
    def survived_percentage(self):
        return '{0:.2f}'.format(self._percents('Survived')[1])

    @property
    def first_class_percentage(self):
        return '{0:.2f}'.format(self._percents('Pclass')[1])

    @property
    def std_and_mean_for_age(self):
        return '{0:.2f} {1:.2f}'.format(
            self.titanic_data['Age'].std(),
            self.titanic_data['Age'].mean()
        )

    @property
    def correlation_sibsp_parch(self):
        return '{0:.2f}'.format(
                self.titanic_data['SibSp'].corr(self.titanic_data['Parch'])
        )

    def _get_first_name(self, full_name):
        try:
            return full_name.split('(')[1].replace(')', ' ').split(' ')[0].replace('"', '')

        except IndexError:
            try:
                return full_name.split('Miss. ')[1].split(' ')[0].replace('"', '')

            except IndexError:
                try:
                    return full_name.split('Mrs. ')[1].split(' ')[0].replace('"', '')

                except IndexError:
                    return None

    @property
    def most_popular_female_name(self):
        return self.titanic_data.groupby(
                self.titanic_data[self.titanic_data.Sex == 'female']['Name'].apply(self._get_first_name)
            ).count().idxmax().Name
