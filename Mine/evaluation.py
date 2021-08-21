import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


def get_label_frequency(ontology):
    col_sums = ontology.sum(0)
    index_11_30 = np.where((col_sums >= 11) & (col_sums <= 30))[0]
    index_31_100 = np.where((col_sums >= 31) & (col_sums <= 100))[0]
    index_101_300 = np.where((col_sums >= 101) & (col_sums <= 300))[0]
    index_larger_300 = np.where(col_sums >= 301)[0]
    return index_11_30, index_31_100, index_101_300, index_larger_300


def calculate_accuracy(y_test, y_score):
    y_score_max = np.argmax(y_score, axis=1)
    cnt = 0
    for i in range(y_score.shape[0]):
        if y_test[i, y_score_max[i]] == 1:
            cnt += 1

    return float(cnt) / y_score.shape[0]


def calculate_fmax(preds, labels):
    preds = np.round(preds, 2)
    labels = labels.astype(np.int32)
    f_max = 0
    p_max = 0
    r_max = 0
    sp_max = 0
    t_max = 0
    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (preds > threshold).astype(np.int32)
        tp = np.sum(predictions * labels)
        fp = np.sum(predictions) - tp
        fn = np.sum(labels) - tp
        sn = tp / (1.0 * np.sum(labels))
        sp = np.sum((predictions ^ 1) * (labels ^ 1))
        sp /= 1.0 * np.sum(labels ^ 1)
        fpr = 1 - sp
        precision = tp / (1.0 * (tp + fp))
        recall = tp / (1.0 * (tp + fn))
        f = 2 * precision * recall / (precision + recall)
        if f_max < f:
            f_max = f
            p_max = precision
            r_max = recall
            sp_max = sp
            t_max = threshold
    return f_max


def plot_PRCurve(label, score):
    precision, recall, _ = precision_recall_curve(label.ravel(), score.ravel())
    aupr = average_precision_score(label, score, average="micro")

    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b',
                     **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(aupr))
    plt.show()


def evaluate_performance(y_test, y_score):
    """Evaluate performance"""
    n_classes = y_test.shape[1]
    # n_classes = 2
    perf = dict()

    perf["M-aupr"] = 0.0
    n = 0
    aupr_list = []
    num_pos_list = []
    for i in range(n_classes):
        num_pos = sum(y_test[:, i])
        if num_pos > 0:
            ap = average_precision_score(y_test[:, i], y_score[:, i])
            n += 1
            perf["M-aupr"] += ap
            aupr_list.append(ap)
            num_pos_list.append(num_pos)
    perf["M-aupr"] /= n
    perf['aupr_list'] = aupr_list
    perf['num_pos_list'] = num_pos_list

    # Compute micro-averaged AUPR
    perf['m-aupr'] = average_precision_score(y_test.ravel(), y_score.ravel())

    # Computes accuracy
    perf['accuracy'] = calculate_accuracy(y_test, y_score)

    # # Computes auc
    # fpr, tpr, thresholds = roc_curve(y_test, y_score)
    # perf['auc'] = auc(fpr, tpr)

    # Computes F1-score
    alpha = 3
    y_new_pred = np.zeros_like(y_test)
    for i in range(y_test.shape[0]):
        top_alpha = np.argsort(y_score[i, :])[-alpha:]
        y_new_pred[i, top_alpha] = np.array(alpha * [1])
    perf["F1-score"] = f1_score(y_test, y_new_pred, average='micro')

    perf['F-max'] = calculate_fmax(y_score, y_test)

    return perf


def evaluate_performance_two(y_test, y_pred, y_score):
    """Evaluate performance"""

    sort_y_score = sorted(enumerate(y_score), key=lambda x: x[1], reverse=True)[:100]
    top_y_score = []
    top_y_score_index = []
    for x, y in sort_y_score:
        top_y_score_index.append(x)
        top_y_score.append(y)
    top_y_perd = [int(item > 0.5) for item in top_y_score]
    top_y_test = [y_test[item] for item in top_y_score_index]

    perf = dict()

    perf["ppv"] = precision_score(top_y_test, top_y_perd)

    # Computes accuracy
    perf['accuracy'] = accuracy_score(y_test, y_pred)

    # Computes precision_score
    perf['precision'] = precision_score(y_test, y_pred)

    # Computes recall_score
    perf['recall'] = recall_score(y_test, y_pred)

    # Compute micro-averaged AUPR
    perf['aupr'] = average_precision_score(y_test, y_pred, average='macro')

    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    perf['AUCPR'] = auc(recall, precision)

    # Computes auc
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    perf['auc'] = auc(fpr, tpr)

    # Computes f1-score
    perf["f1-score"] = f1_score(y_test, y_pred)

    return perf


def get_max_avg_results(perf_list):
    perf_max = dict()
    perf_avg = dict()

    ppv_list = []
    accuracy_list = []
    precision_list = []
    recall_list = []
    aupr_list = []
    auc_list = []
    f1_list = []
    AUCPR_list = []
    for perf in perf_list:
        ppv_list.append(perf["ppv"])
        accuracy_list.append(perf["accuracy"])
        precision_list.append(perf["precision"])
        recall_list.append(perf["recall"])
        aupr_list.append(perf["aupr"])
        auc_list.append(perf["auc"])
        f1_list.append(perf["f1-score"])
        AUCPR_list.append((perf["AUCPR"]))
    # Computes max avg perf
    perf_max['ppv'] = max(ppv_list)
    perf_max['accuracy'] = max(accuracy_list)
    perf_max['precision'] = max(precision_list)
    perf_max['recall'] = max(recall_list)
    perf_max['aupr'] = max(aupr_list)
    perf_max['auc'] = max(auc_list)
    perf_max["f1-score"] = max(f1_list)
    perf_max["AUCPR"] = max(AUCPR_list)

    perf_avg['ppv'] = sum(ppv_list) / len(ppv_list)
    perf_avg['accuracy'] = sum(accuracy_list) / len(accuracy_list)
    perf_avg['precision'] = sum(precision_list) / len(precision_list)
    perf_avg['recall'] = sum(recall_list) / len(recall_list)
    perf_avg['aupr'] = sum(aupr_list) / len(aupr_list)
    perf_avg['auc'] = sum(auc_list) / len(auc_list)
    perf_avg["f1-score"] = sum(f1_list) / len(f1_list)
    perf_avg["AUCPR"] = sum(AUCPR_list) / len(AUCPR_list)

    perf_result = dict()
    perf_result['max'] = perf_max
    perf_result['average'] = perf_avg
    return perf_result


def get_results(ontology, Y_test, y_score):
    perf = defaultdict(dict)
    perf['all'] = evaluate_performance(Y_test, y_score)

    return perf


def get_metrics(real_score, predict_score):
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

    recall_list = tpr
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]

    return [aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision]


def cv_model_evaluate(interaction_matrix, predict_matrix, train_matrix):
    test_index = np.where(train_matrix == 0)
    real_score = interaction_matrix[test_index]
    predict_score = predict_matrix[test_index]
    return get_metrics(real_score, predict_score)