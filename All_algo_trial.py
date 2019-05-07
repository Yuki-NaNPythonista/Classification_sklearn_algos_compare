# coding=utf-8

"""
Try the classification algorithm divided into two classes and list each acuracy.
Plot with matplotlib.

List of available algorithms:
    *lr:logistic_regression
    *svc:SVC
    *nb:GaussianNB

hyper parameter settings:
    *logistic_regression : lr = {hyper parameter_name:value}
    ...
    if none is default!
"""

from sklearn import datasets

from Classification_sklearn_algos_conpare.algos import Lr as Lr
from Classification_sklearn_algos_conpare.algos import Svc as Svc
from Classification_sklearn_algos_conpare.algos import Nb as Nb
from Classification_sklearn_algos_conpare.algos import Sgd as Sgd

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

import pandas as pd


class SomeClassificationChecker:

    # Stratified K-Fold CV で性能を評価する
    skf = StratifiedKFold(shuffle=True)
    algo_list = {"lr": None,
                 "svc": None,
                 "nb": None,
                 "sgd": None}

    def __init__(self, **kwargs):

        if len(kwargs.items()) == 0:
            self.algo_list["lr"] = Lr.logistic_regression_model()
            self.algo_list["svc"] = Svc.svc_model()
            self.algo_list["nb"] = Nb.naive_bayes_model()
            self.algo_list["sgd"] = Sgd.sgd_model()
            return

        for key, value in kwargs.items():
            if "lr" in key:
                self.algo_list["lr"] = Lr.logistic_regression_model(lr_dict=value)
            elif "svc" in key:
                self.algo_list["svc"] = Svc.svc_model(svc_dict=value)
            elif "nb" in key:
                self.algo_list["nb"] = Nb.naive_bayes_model(nb_dict=value)
            elif "sgd" in key:
                self.algo_list["sgd"] = Sgd.sgd_model(sgd_dict=value)

    def algo_check(self, work_data, work_label):
        score_data = pd.DataFrame()

        scoring = {
            'acc': 'accuracy',
            'auc': 'roc_auc',
        }

        for key, value in self.algo_list.items():
            scores = cross_validate(value, work_data, work_label, cv=self.skf, scoring=scoring)

            score_data[f"{key}_acc"] = scores['test_acc']
            score_data[f"{key}_auc"] = scores['test_auc']

            print(key + "'s Accuracy (mean):", scores['test_acc'].mean())
            print(key + "'s AUC (mean):", scores['test_auc'].mean())

        return score_data


def main():
    scc = SomeClassificationChecker()
    scc.algo_check()


if __name__ == '__main__':
    main()
    

