def conf_matrix(self, labels_test, labels_test_predictions):
    """
    Calculates the different values of the confusion matrix.

    ----------
    Input: conf_matrix(model, labels_values, labels_predicted_values)
    ----------
    Output: cm = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    """
    matrix = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    for index, label in enumerate(labels_test):
        predictions = labels_test_predictions[:, 1][index]
        if label == 1:
            if label == predictions:
                cm['TP'] += 1
            else:
                cm['FN'] += 1
        else:
            if label == predictions:
                cm['TN'] += 1
            else:
                cm['FP'] += 1
        self.cm_values = matrix
    return matrix
