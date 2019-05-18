# coding=utf-8
"""
Random Forest classifier.
"""
from sklearn.ensemble import RandomForestClassifier


class Rf:
    def __init__(self):
        pass

    @staticmethod
    def random_forest_model(rf_dict=None):
        if rf_dict is None:
            rf = RandomForestClassifier()
        else:
            hp_dict = {"n_estimators": 'warn',
                       "criterion": 'gini',
                       "max_depth": None,
                       "min_samples_split": 2,
                       "min_samples_leaf": 1,
                       "min_weight_fraction_leaf": 0.0,
                       "max_features": "auto",
                       "max_leaf_nodes": None,
                       "min_impurity_decrease": 0.0,
                       "min_impurity_split": None,
                       "bootstrap": True,
                       "oob_score": False,
                       "n_jobs": None,
                       "random_state": None,
                       "verbose": 0,
                       "warm_start": False,
                       "class_weight": None
                       }

            for para_key, para_value in rf_dict:
                hp_dict[para_key] = para_value

            rf = RandomForestClassifier(n_estimators='warn',
                                        criterion='gini',
                                        max_depth=None,
                                        min_samples_split=2,
                                        min_samples_leaf=1,
                                        min_weight_fraction_leaf=0.0,
                                        max_features="auto",
                                        max_leaf_nodes=None,
                                        min_impurity_decrease=0.0,
                                        min_impurity_split=None,
                                        bootstrap=True,
                                        oob_score=False,
                                        n_jobs=None,
                                        random_state=None,
                                        verbose=0,
                                        warm_start=False,
                                        class_weight=None)

        return rf
