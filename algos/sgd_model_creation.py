# coding=utf-8
"""
stochastic gradient descent (SGD) learning
"""
from sklearn.linear_model import SGDClassifier


class Sgd:
    def __init__(self):
        pass

    @staticmethod
    def sgd_model(sgd_dict=None):
        if sgd_dict is None:
            sgd = SGDClassifier()
        else:
            hp_dict = {"loss": "hinge",
                       "penalty": "l2",
                       "alpha": 0.0001,
                       "l1_ratio": 0.15,
                       "fit_intercept": True,
                       "max_iter": None,
                       "tol": None,
                       "shuffle": True,
                       "verbose": 0,
                       "epsilon": 0.1,
                       "n_jobs": None,
                       "random_state": None,
                       "learning_rate": "optimal",
                       "eta0": 0.0,
                       "power_t": 0.5,
                       "early_stopping": False,
                       "validation_fraction": 0.1,
                       "n_iter_no_change": 5,
                       "class_weight": None,
                       "warm_start": False,
                       "average": False,
                       "n_iter": None}

            for para_key, para_value in sgd_dict:
                hp_dict[para_key] = para_value

            sgd = SGDClassifier(loss=hp_dict["loss"],
                                penalty=hp_dict["penalty"],
                                alpha=hp_dict["alpha"],
                                l1_ratio=hp_dict["l1_ratio"],
                                fit_intercept=hp_dict["fit_intercept"],
                                max_iter=hp_dict["max_iter"],
                                tol=hp_dict["tol"],
                                shuffle=hp_dict["shuffle"],
                                verbose=hp_dict["verbose"],
                                epsilon=hp_dict["epsilon"],
                                n_jobs=hp_dict["n_jobs"],
                                random_state=hp_dict["random_state"],
                                learning_rate=hp_dict["learning_rate"],
                                eta0=hp_dict["eta0"],
                                power_t=hp_dict["power_t"],
                                early_stopping=hp_dict["early_stopping"],
                                validation_fraction=hp_dict["validation_fraction"],
                                n_iter_no_change=hp_dict["n_iter_no_change"],
                                class_weight=hp_dict["class_weight"],
                                warm_start=hp_dict["warm_start"],
                                average=hp_dict["average"],
                                n_iter=hp_dict["n_iter"])

        return sgd

