# coding=utf-8
"""
C-Support Vector Classification.
"""

from sklearn.svm import SVC


class Svc:
    def __init__(self):
        pass

    @staticmethod
    def svc_model(svc_dict=None):

        if svc_dict is None:
            svc = SVC(kernel='linear', random_state=None)
        else:
            hp_dict = {"C": 1.0,
                       "kernel": 'rbf',
                       "degree": 3,
                       "gamma": 'auto_deprecated',
                       "coef0": 0.0,
                       "shrinking": True,
                       "probability": False,
                       "tol": 0.001,
                       "cache_size": 200,
                       "class_weight": None,
                       "verbose": False,
                       "max_iter": -1,
                       "decision_function_shape": 'ovr',
                       "random_state": None}

            for para_key, para_value in svc_dict:
                hp_dict[para_key] = para_value

            svc = SVC(C=hp_dict["C"],
                      kernel=hp_dict["kernel"],
                      degree=hp_dict["degree"],
                      gamma=hp_dict["gamma"],
                      coef0=hp_dict["coef0"],
                      shrinking=hp_dict["shrinking"],
                      probability=hp_dict["probability"],
                      tol=hp_dict["tol"],
                      cache_size=hp_dict["cache_size"],
                      class_weight=hp_dict["class_weight"],
                      verbose=hp_dict["verbose"],
                      max_iter=hp_dict["max_iter"],
                      decision_function_shape=hp_dict["decision_function_shape"],
                      random_state=hp_dict["random_state"])

        return svc
