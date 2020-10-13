import pandas as pd


class GetScore:
    """
    Calculates the score for a given model.
    Available methods: precision_score(), recall_score(), f1_score(), auc_score()
    Models need to be defined externally.

    ----------
    Input: method(labels, features, model)
    ----------
    Output: Score requested
    """

    def __init__(self, features, labels, model):
        self.features = features
        self.labels = labels
        self.model = model
        self.label_hat = model.predict(features)

    @staticmethod
    def precision_score(labels, features, model):
        label_hat = labels_hat(features, model)
        labels_labels_hat = list(zip(labels, label_hat))
        tp = sum([1 for i in labels_labels_hat if i[0] == 1 and i[1] == 1])
        fp = sum([1 for i in labels_labels_hat if i[0] == 0 and i[1] == 1])
        return tp / float(tp + fp)

    @staticmethod
    def recall_score(labels, features, model):
        label_hat = labels_hat(features, model)
        labels_labels_hat = list(zip(labels, label_hat))
        tp = sum([1 for i in labels_labels_hat if i[0] == 1 and i[1] == 1])
        fn = sum([1 for i in labels_labels_hat if i[0] == 1 and i[1] == 0])
        return tp / float(tp + fn)

    @staticmethod
    def f1_score(labels, features, model):
        label_hat = labels_hat(features, model)
        precision_score = precision(labels, label_hat)
        recall_score = recall(labels, label_hat)
        return 2 * ((precision_score * recall_score) / (precision_score + recall_score))

    @staticmethod
    def auc_score(labels, features, model):
        labels_train_predictions = model.predict(features)
        fpr, tpr, thresholds = roc_curve(labels, labels_train_predictions)
        return auc(fpr, tpr)
