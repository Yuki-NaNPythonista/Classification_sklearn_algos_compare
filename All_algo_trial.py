# coding=utf-8

'''
Try the classification algorithm divided into two classes and list each acuracy.
Plot with matplotlib.
'''

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

# Stratified K-Fold CV で性能を評価する
skf = StratifiedKFold(shuffle=True)


def logistic_regression_model(work_data, work_label):
    # ロジスティック回帰
    clf = LogisticRegression()

    scoring = {
        'acc': 'accuracy',
        'auc': 'roc_auc',
    }

    scores = cross_validate(clf, work_data, work_label, cv=skf, scoring=scoring)

    print('Accuracy (mean):', scores['test_acc'].mean())
    print('AUC (mean):', scores['test_auc'].mean())




def main():
    pass


if __name__ == '__main__':
    main()
    

