import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, matthews_corrcoef, balanced_accuracy_score, roc_curve

import logging
logger = logging.getLogger(__name__)


def compute_perf_metrics(y_true, y_pred_proba, positive_label, threshold=0.5, sample_weight=None):
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)

    if sample_weight is not None:
        logger.info("Using sample_weight to evaluate performances")

    y_pred = y_pred_proba >= threshold

    # Calc all metrics
    recall = recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0, sample_weight=sample_weight)

    if len(np.unique(y_true)) == 2:
        precision = precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0, sample_weight=sample_weight)
        roc_auc = roc_auc_score(y_true, y_pred_proba, sample_weight=sample_weight)
        mcc = matthews_corrcoef(y_true, y_pred, sample_weight=sample_weight)
        ba = balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weight)
        # Evaluate optimal threshold (to be used when testing model by using it as arg of this method)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba, sample_weight=sample_weight)
        optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    else:
        precision = roc_auc = mcc = ba = optimal_threshold = float("NaN")

    return {"optimal_threshold": optimal_threshold, "precision": precision, "recall": recall, "roc_auc": roc_auc, "mcc": mcc, "balanced_acc": ba}


def apply_model(model, y_test_samples, sample_to_weight, metapath_dict):
    '''
    Wrapper method applying the model to samples and returning all sort of results
    Refer to individual method documentation for more details
    '''

    positive_samples = y_test_samples[y_test_samples == 1].index
    negative_samples = y_test_samples[y_test_samples == 0].index

    sample_to_class = {}
    for positive in positive_samples:
        sample_to_class[positive] = 1
    for negative in negative_samples:
        sample_to_class[negative] = 0

    sample_class_list = list(sample_to_class.items())
    samples = [s for s,c in sample_class_list]
    y_test = [c for s,c in sample_class_list]
    sample_weights = [sample_to_weight[s] for s,c in sample_class_list]

    ordered_samples, predict_probas, explanations = model.predict_and_explain(samples, metapath_dict)

    predict_positive_probas = predict_probas[:,1]
    perf_metrics = compute_perf_metrics(y_test, predict_positive_probas, positive_label=1, sample_weight=sample_weights)
    return ordered_samples, sample_weights, predict_positive_probas, y_test, perf_metrics, explanations
