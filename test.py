# coding=utf-8
from Classification_sklearn_algos_conpare.All_algo_trial import *
import pandas as pd


def main():
    # data prepare
    test_data_path = r'./test_data/train.csv'
    test_data = pd.read_csv(test_data_path)
    test_data = test_data.dropna()
    test_data = pd.get_dummies(test_data, columns=['Embarked'])
    test_data = pd.get_dummies(test_data, columns=["Sex"])
    test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean())
    test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
    work_data = test_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
    work_label = test_data["Survived"]

    checker = SomeClassificationChecker()
    # checker = SomeClassificationChecker(lr={"tol": 0.0001},
    #                                     svc={"degree": 5},
    #                                     nb={"var_smoothing": 1e-08},
    #                                     sgd={"alpha": 0.0005})

    checker.algo_check(work_data, work_label)


if __name__ == '__main__':
    main()
    

