# coding=utf-8
"""
Gaussian Naive Bayes
"""
from sklearn.naive_bayes import GaussianNB


class Nb:
    def __init__(self):
        pass

    @staticmethod
    def naive_bayes_model(nb_dict=None):

        if nb_dict is None:
            nv = GaussianNB()
        else:
            hp_dict = {"priors": None,
                       "var_smoothing": 1e-09}

            for para_key, para_value in nb_dict:
                hp_dict[para_key] = para_value

            nv = GaussianNB(priors=hp_dict["priors"],
                            var_smoothing=hp_dict["var_smoothing"])

        return nv

