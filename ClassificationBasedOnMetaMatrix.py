#!/usr/bin/python

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import EditedNearestNeighbours

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN

from imblearn.ensemble import EasyEnsemble
from imblearn.ensemble import BalanceCascade

from sklearn.metrics import matthews_corrcoef


def LoadDataset(file):
    data = pd.read_csv(file)
    X, y = np.array(data.ix[:, 1:]), np.array(data.ix[:, 0])
    return X, y


def LoadDatasetX(file):
    data = pd.read_csv(file)
    return np.array(data)


def LoadDatasety(file):
    data = pd.read_csv(file)
    return np.array(data)


# Feature Selection
def JmimTransform(X, y, method='JMIM'):
    feature_selector = mifs.MutualInformationFeatureSelector(method=method)
    feature_selector.fit(X, y)
    X_filtered = X.ix[:, feature_selector.ranking_]
    return X_filtered


# Algorithms of classification
# 1. Multinoimal Naive Bayes
def naive_bayes_classifier(X, y, classes=2):
    from sklearn.naive_bayes import MultinomialNB
    if classes == 2:
        model = MultinomialNB(alpha=0.01).fit(X, y)
    else:
        model = OneVsRestClassifier(MultinomialNB(alpha=0.01)).fit(X, y)
    return model


# 2. KNN classifer
def knn_classifier(X, y, classes=2):
    from sklearn.neighbors import KNeighborsClassifier
    if classes == 2:
        model = KNeighborsClassifier().fit(X, y)
    else:
        model = OneVsRestClassifier(KNeighborsClassifier()).fit(X, y)
    return model


# 3. Logistic Regression Classifier
def logistic_regression_classifier(X, y, classes=2):
    from sklearn.linear_model import LogisticRegression
    if classes == 2:
        model = LogisticRegression(penalty='l2').fit(X, y)
    else:
        model = OneVsRestClassifier(LogisticRegression(penalty='l2')).fit(X, y)
    return model


# 4.Random Forest Classifier
def random_forest_classifier(X, y, classes=2):
    from sklearn.ensemble import RandomForestClassifier
    if classes == 2:
        model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    else:
        model = OneVsRestClassifier(RandomForestClassifier(n_estimators=10, random_state=42)).fit(X, y)
    return model


# 5. Decision Tree Classifier
def decision_tree_classifer(X, y, classes=2):
    from sklearn import tree
    if classes == 2:
        model = tree.DecisionTreeClassifier().fit(X, y)
    else:
        model = OneVsRestClassifier(tree.DecisionTreeClassifier()).fit(X, y)
    return model


# 6.GBDT(Gradient Boosting Decision Tree)
def gradient_boosting_classifier(X, y, classes=2):
    from sklearn.ensemble import GradientBoostingClassifier
    if classes == 2:
        model = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X, y)
    else:
        model = OneVsRestClassifier(GradientBoostingClassifier(n_estimators=100, random_state=42)).fit(X, y)
    return model


# 7.SVM Classifier
def svm_classifier(X, y, classes=2):
    from sklearn.svm import SVC
    if classes == 2:
        model = SVC(kernel='linear', probability=True).fit(X, y)
    else:
        model = OneVsRestClassifier(SVC(kernel='linear', probability=True)).fit(X, y)
    return model


# 8.Stochastic Gradient Descent (SGD)
def sgd_classifier(X, y, classes=2):
    from sklearn.linear_model import SGDClassifier
    if classes == 2:
        model = SGDClassifier(random_state=42).fit(X, y)
    else:
        model = OneVsRestClassifier(SGDClassifier(loss='log', penalty='l2', random_state=42)).fit(X, y)
    return model


# AdaBoost Classifier
def adaboost_classifier(X, y, classes=2):
    from sklearn.ensemble import AdaBoostClassifier
    if classes == 2:
        model = AdaBoostClassifier().fit(X, y)
    else:
        model = OneVsRestClassifier(AdaBoostClassifier()).fit(X, y)
    return model


# Neural networks Classifier
def neural_classifier(X, y, classes=2):
    from sknn.mlp import Classifier, Layer
    nn = Classifier(layers=[Layer("Rectifier", units=100),Layer("Softmax")],
                    learning_rate=0.02,n_iter=10)
    if classes == 2:
        model = nn.fit(X, y)
    else:
        model = OneVsRestClassifier(nn).fit(X, y)
    return model


# Generate the Reciver Operating Curve(ROC)
def rocPlot(fprs, tprs, ofp):
    n = len(fprs)
    plt.figure()
    if n == 1:
        plt.plot(fprs[0], tprs[0], label='ROC curve (area={0:0.2f})'
                 ''.format(auc(fprs[0], tprs[0])))
    else:
        for i in range(n):
            plt.plot(fprs[i], tprs[i], label='ROC curve of Class{0}(area={1:0.2f})'
                     ''.format(i, auc(fprs[i], tprs[i])))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.title('Reciver Operating Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(ofp,format = 'tiff', dpi=300)
    plt.close()


#MCC for multi class
def mcc_multiclass(y_true, y_predict):
    rows, cols = y_true.shape
    xy = 0
    xx = 0
    yy = 0
    for i in range(cols):
        xx += np.cov(np.nan_to_num(y_true[:, i]), np.nan_to_num(y_predict[:, i]))[0][0]
        xy += np.cov(np.nan_to_num(y_true[:, i]), np.nan_to_num(y_predict[:, i]))[0][1]
        yy += np.cov(np.nan_to_num(y_true[:, i]), np.nan_to_num(y_predict[:, i]))[1][1]
    mcc = xy / (np.sqrt(np.abs(xx) * np.abs(yy)) * np.sign(xx) * np.sign(yy))
    return mcc


# binary or multiclass classification
def classification_pipe(X, y):

    f1_classifer = []
    mcc_classifer = []
    aucs = []
    classifiers = {'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                   'DT': decision_tree_classifer,
                   'SVM': svm_classifier,
                   'GBDT': gradient_boosting_classifier,
                   'SGD': sgd_classifier,
                   'ADA': adaboost_classifier,
                    }
#   classifiers = {'NB': naive_bayes_classifier,
#                  'KNN': knn_classifier,
#                   'LR': logistic_regression_classifier,
#                   'RF': random_forest_classifier,
#                   'DT': decision_tree_classifer,
#                   'SVM': svm_classifier,
#                   'GBDT': gradient_boosting_classifier,
#                   'SGD': sgd_classifier,
#                   'ADA': adaboost_classifier,
#                   'NN': neural_classifier
#                   }

    is_binary_class = (len(np.unique(y)) == 2)
    classes = len(np.unique(y))
    test_classifier = ['KNN', 'LR', 'RF', 'DT', 'SVM', 'GBDT', 'SGD', 'ADA']
    x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.4, random_state=42)
    flag = 0
    if is_binary_class:
        for classifier in test_classifier:
            flag += 1
            fprs = []
            tprs = []
            model = classifiers[classifier](x_train, y_train, classes=classes)
            predict = model.predict(x_test)
            fpr, tpr, _ = roc_curve(y_test, predict)
            aucs.append(auc(fpr, tpr))
            f1_classifer.append(f1_score(y_test, predict))
            mcc_classifer.append(matthews_corrcoef(y_test, predict))
            fprs.append(fpr)
            tprs.append(tpr)
            rocPlot(fprs, tprs, '/Users/user/project/PhD/data/rocplot_%s.png'
                    % classifier)
    else:
        for classifier in test_classifier:
            flag += 1
            fprs = []
            tprs = []
            y_proba = classifiers[classifier](x_train, y_train, classes=classes).predict_proba(x_test)
            y_predict = classifiers[classifier](x_train, y_train, classes=classes).predict(x_test)
            f1_value = f1_score(y_test, y_predict, average='weighted')
            f1_classifer.append(f1_value)
            y_test_bin = label_binarize(y_test, classes=np.unique(y))
            mcc_classifer.append(mcc_multiclass(y_test_bin, y_proba))
            tmp_auc = []
            for i in range(classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                fprs.append(fpr)
                tprs.append(tpr)
                tmp_auc.append(auc(fpr,tpr))
            aucs.append(np.mean(tmp_auc))
            rocPlot(fprs, tprs, '/Users/user/project/PhD/data/rocplotm_%s.tiff'
                    % classifier)
    return f1_classifer, mcc_classifer, aucs


# main function
def main():

    # load dataset
    ipf = sys.argv[1]
    X, y = LoadDataset(ipf)

    # Under Sampling to tackle the unbalanced dataset between multi-class
    verbose = False
    ratio = 0.9
#    nm3 = NearMiss(version=3, ratio=ratio, random_state=42)
#    nm3x1, nm3y1 = nm3.fit_sample(X, y)
    SENN = SMOTEENN(ratio=ratio, verbose=verbose, random_state=42)
    nm3x2, nm3y2 = SENN.fit_sample(X, y)
#    for i in range(100, 502, 20):
#        subx = nm3x1[:, :i+1]
#        f1_val, mcc_val, auc = classification_pipe(subx, nm3y1)
#        print "NM " + str(i) + ' F1 ' + str(f1_val)
    for i in range(300, 301):
        subx = nm3x2[:, :i+1]
        f1_val, mcc_val, auc = classification_pipe(subx, nm3y2)
        print "SENN " + str(i) + ' F1 ' + str(f1_val)

# test
if __name__ == '__main__':
    main()

