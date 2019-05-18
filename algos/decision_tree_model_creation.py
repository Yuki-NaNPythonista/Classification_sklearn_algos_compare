# coding=utf-8
"""
Decision Tree classifier.
"""
from sklearn import tree


class Dt:
    def __init__(self):
        pass

    @staticmethod
    def decision_tree_model(dt_dict=None):
        if dt_dict is None:
            dt = tree.DecisionTreeClassifier()
        else:
            hp_dict = {"criterion": "gini",
                       "splitter": 'best',
                       "max_depth": None,
                       "min_samples_split": 2,
                       "min_samples_leaf": 1,
                       "min_weight_fraction_leaf": 0.0,
                       "max_features": None,
                       "random_state": None,
                       "max_leaf_nodes": None,
                       "min_impurity_decrease": 0.0,
                       "min_impurity_split": None,
                       "class_weight": None,
                       "presort": False
                       }

            for para_key, para_value in dt_dict:
                hp_dict[para_key] = para_value

            dt = tree.DecisionTreeClassifier(criterion=hp_dict["criterion"],
                                             splitter=hp_dict["splitter"],
                                             max_depth=hp_dict["max_depth"],
                                             min_samples_split=hp_dict["min_samples_split"],
                                             min_samples_leaf=hp_dict["min_samples_leaf"],
                                             min_weight_fraction_leaf=hp_dict["min_weight_fraction_leaf"],
                                             max_features=hp_dict["max_features"],
                                             random_state=hp_dict["random_state"],
                                             max_leaf_nodes=hp_dict["max_leaf_nodes"],
                                             min_impurity_decrease=hp_dict["min_impurity_decrease"],
                                             min_impurity_split=hp_dict["min_impurity_split"],
                                             class_weight=hp_dict["class_weight"],
                                             presort=hp_dict["presort"]
                                             )

        return dt

