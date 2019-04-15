# coding=utf-8
"""
Logistic Regression (aka logit, MaxEnt) classifier.
"""
from sklearn.linear_model import LogisticRegression


class Lr:
    def __init__(self):
        pass

    @staticmethod
    def logistic_regression_model(lr_dict=None):
        if lr_dict is None:
            lr = LogisticRegression()
        else:
            hp_dict = {"penalty": 'l2',
                       "dual": False,
                       "tol": 0.0001,
                       "C": 1.0,
                       "fit_intercept": True,
                       "intercept_scaling": 1,
                       "class_weight": None,
                       "random_state": None,
                       "solver": 'warn',
                       "max_iter": 100,
                       "multi_class": 'warn',
                       "verbose": 0,
                       "warm_start": False,
                       "n_jobs": None}

            for para_key, para_value in lr_dict:
                hp_dict[para_key] = para_value

            lr = LogisticRegression(penalty=hp_dict["penalty"],
                                    dual=hp_dict["dual"],
                                    tol=hp_dict["tol"],
                                    C=hp_dict["C"],
                                    fit_intercept=hp_dict["fit_intercept"],
                                    intercept_scaling=hp_dict["intercept_scaling"],
                                    class_weight=hp_dict["class_weight"],
                                    random_state=hp_dict["random_state"],
                                    solver=hp_dict["solver"],
                                    max_iter=hp_dict["max_iter"],
                                    multi_class=hp_dict["multi_class"],
                                    verbose=hp_dict["verbose"],
                                    warm_start=hp_dict["warm_start"],
                                    n_jobs=hp_dict["n_jobs"])

        return lr

