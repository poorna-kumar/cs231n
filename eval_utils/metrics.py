from sklearn.metrics import coverage_error, roc_auc_score
import numpy as np


def coverage_err(y_true, y_pred):
    """
    Coverage error:
    For every sample, how far down the ranked list of predicted classes must we reach to get all
    actual class labels? The average value of this metric across samples is the coverage error.

    :param y_true: array of shape (n_samples, n_labels)
    :param y_pred: array of shape (n_samples, n_labels)
    :return: coverage_error, float
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if any(np.sum(y_true, axis=1) == 0):
        raise ValueError(
            "Every sample must correspond to at least one positive label."
            "No row of y_true can sum to 0."
        )
    return coverage_error(y_true, y_pred)


def avg_auc_macro(y_true, y_pred):
    """
    Average AUC:
    For each class, compute the AUC separately. Return average AUC across individual class AUCs.

    :param y_true: array of shape (n_samples, n_labels)
    :param y_pred: array of shape (n_samples, n_labels)
    :return: avg_auc, float
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return roc_auc_score(y_true, y_pred, average="macro")


def avg_auc_micro(y_true, y_pred):
    """
    Average AUC:
    Treat each predicted value and label as one of FP, TN, TP, FN. Find AUC across all.

    :param y_true:
    :param y_pred:
    :return:
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return roc_auc_score(y_true, y_pred, average="micro")
