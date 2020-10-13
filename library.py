#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import auc
from sklearn.metrics import roc_curve

#####################
#  Visualisations   #
################################################################################


def word_dist_plot(dataframe: pd.core.frame.DataFrame,
                   start: int = 0,
                   end: int = -1
                   ) -> pd.core.frame.DataFrame or int or int:
    """
    Creates distplot for text columns.
    It will calculate the length of the documents
    within these columns and represent the population
    of fraudulent and legit job ads for those columns.

    NOTE: the y axis will be truncated at 800 to provide
    a better visual of the fraudulent ads population.

    ----------
    Input: word_dist_plot(dataframe[, start_of_column_slicing=0, end_of_column_slicing=-1])
    ----------
    Output: one distplot per column
    """
    if isinstance(dataframe, pd.core.frame.DataFrame):
        for column in dataframe.columns[start:end]:
            legit_desc_words = dataframe[dataframe['fraudulent'] == 0][column].str.len(
            )
            fraud_desc_words = dataframe[dataframe['fraudulent'] == 1][column].str.len(
            )
            sns.distplot(legit_desc_words,
                         kde=False,
                         label='Legitimate job posts',
                         color=sns.color_palette('RdYlGn')[4],
                         hist_kws={"alpha": 1})
            sns.distplot(fraud_desc_words,
                         kde=False,
                         label='Fraudulent job posts',
                         color=sns.color_palette('RdYlGn')[1],
                         hist_kws={"alpha": 1})
            plt.xlabel('Word count')
            plt.ylabel('Number of posts')
            plt.ylim(0, 800)
            plt.title(f"How many words have been used in the job's {column}?")
            plt.legend()
            plt.show()
    else:
        raise TypeError('arguments must be pandas.DataFrame, int, int')


################################################################################


def count_plot(dataframe: pd.core.frame.DataFrame,
               start: int = 0,
               end: int = -1
               ) -> pd.core.frame.DataFrame or int or int:
    """
    Creates a countplot per categorical columns.
    It differenciates fraudulent and legitimate posts populations.

    ----------
    Input: count_plot(dataframe[, start_of_column_slicing=0, end_of_column_slicing=-1])
    ----------
    Output: one countplot per column
    """
    if isinstance(dataframe, pd.core.frame.DataFrame):
        for column in dataframe.columns[start:end]:
            sns.countplot(x=column, hue='fraudulent',
                          palette='RdYlGn_r', data=dataframe, saturation=1)
            plt.title(f'Is there a type of {column} targeted by fraudsters?')
            plt.xticks(rotation=90)
            plt.ylim(0, 1000)
            plt.legend()
            plt.show()
    else:
        raise TypeError('arguments must be pandas.DataFrame, int, int')


################################################################################


def conf_matrix(self, labels_test, labels_test_predictions):
    """
    Calculates the different values of the confusion matrix.

    ----------
    Input: conf_matrix(model, labels_values, labels_predicted_values)
    ----------
    Output: cm = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    """
    cmatrix = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    for index, label in enumerate(labels_test):
        predictions = labels_test_predictions[:,1][index]
        if label == 1:
            if label == pred:
                cm['TP'] += 1
            else:
                cm['FN'] += 1
        else:
            if label == pred:
                cm['TN'] += 1
            else:
                cm['FP'] += 1
        self.cm_values = cmatrix
    return cmatrix


################################################################################


def buildROC(target_train, train_preds, target_test, test_preds):
    """
    Creates the Receiver Operating Characteristic curve.

    ----------
    Input: buildROC(labels_train_values, labels_predicted_train_values,
           labels_test_values, labels_predicted_test_values)
    ----------
    Output: file called 'roc.png'
    """
    fpr, tpr, threshold = roc_curve(target_test, test_preds)
    roc_auc = auc(fpr, tpr)
    fpr1, tpr1, threshold = roc_curve(target_train, train_preds)
    roc_auc1 = auc(fpr1, tpr1)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr1, tpr1, 'b', label='Train AUC = %0.2f' % roc_auc1, color=sns.color_palette()[1])
    plt.plot(fpr, tpr, 'b', label='Validation AUC = %0.2f' % roc_auc, color=sns.color_palette()[4])
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.gcf().savefig('./img/roc.png')


################################################################################


def annot(fpr, tpr, thr):
    """
    Annotates the ROC curve.

    ----------
    Input: annot(FPR, TPR, Threshold) as per calculated by the roc_curve function
           (from sklearn.metrics import roc_curve)
    ----------
    Output: values of the various thresholds along the ROC curve
    """
    k=0
    for i,j in zip(fpr,tpr):
        if k %75 == 0:
            plt.annotate(round(thr[k],2),xy=(i,j), textcoords='data')
        k+=1
