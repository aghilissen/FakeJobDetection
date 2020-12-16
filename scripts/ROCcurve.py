import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


def buildROC(target_train, train_predictions, target_test, test_predictions):
    """
    Creates the Receiver Operating Characteristic curve.

    ----------
    Input: buildROC(labels_train_values, labels_predicted_train_values,
           labels_test_values, labels_predicted_test_values)
    ----------
    Output: file called 'roc.png'
    """
    fpr, tpr, threshold = roc_curve(target_test, test_predictions)
    roc_auc = auc(fpr, tpr)
    fpr1, tpr1, threshold = roc_curve(target_train, train_predictions)
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


def annotate(fpr, tpr, thr):
    """
    Annotates the ROC curve.

    ----------
    Input: annotate(FPR, TPR, Threshold) as per calculated by the roc_curve function
           (from sklearn.metrics import roc_curve)
    ----------
    Output: values of the various thresholds along the ROC curve
    """
    k = 0
    for i, j in zip(fpr, tpr):
        if k % 75 == 0:
            plt.annotate(round(thr[k], 2), xy=(i, j), textcoords='data')
        k += 1
