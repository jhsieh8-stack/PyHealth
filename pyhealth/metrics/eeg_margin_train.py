import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, f1_score
import torch


def eeg_margin_train_fn(y_true_multi: np.ndarray,
                  y_pred_multi: np.ndarray,
                  tnr_for_margintest,
                  probability_list,
                  final_target_list,
                  margin_list):
    """Computes metrics for ranking tasks.

    Args:
        qrels: Ground truth. A dictionary of query ids and their corresponding
            relevance judgements. The relevance judgements are a dictionary of
            document ids and their corresponding relevance scores.
        results: Ranked results. A dictionary of query ids and their corresponding
            document scores. The document scores are a dictionary of document ids and
            their corresponding scores.
        k_values: A list of integers specifying the cutoffs for the metrics.

    Returns:
        A dictionary of metrics and their corresponding values.

    Examples:
        >>> qrels = {
        ...     "q1": {"d1": 1, "d2": 0, "d3": 1},
        ...     "q2": {"d1": 1, "d2": 1, "d3": 0}
        ... }
        >>> results = {
        ...     "q1": {"d1": 0.5, "d2": 0.2, "d3": 0.1},
        ...     "q2": {"d1": 0.1, "d2": 0.2, "d3": 0.5}
        ... }
        >>> k_values = [1, 2]
        >>> ranking_metrics_fn(qrels, results, k_values)
        {'NDCG@1': 0.5, 'MAP@1': 0.25, 'Recall@1': 0.25, 'P@1': 0.5, 'NDCG@2': 0.5, 'MAP@2': 0.375, 'Recall@2': 0.5, 'P@2': 0.5}
    """
    
    y_true_multi = np.concatenate(y_true_multi, 0)
    y_pred_multi = np.concatenate(y_pred_multi, 0)

    auc = roc_auc_score(y_true_multi[:,1], y_pred_multi[:,1])
    apr = average_precision_score(y_true_multi[:,1], y_pred_multi[:,1])
    y_true_multi_array = np.argmax(y_true_multi, axis=1)

    f1 = 0
    for i in range(1, 200):
        threshold = float(i) / 200
        temp_output = np.array(y_pred_multi[:,1])
        temp_output[temp_output>=threshold] = 1
        temp_output[temp_output<threshold] = 0
        temp_score = f1_score(y_true_multi_array, temp_output, average="binary")
        if temp_score > f1:
            f1 = temp_score
        
    result = np.round(np.array([auc, apr, f1]), decimals=4)
    fpr, tpr, thresholds = roc_curve(y_true_multi_array, y_pred_multi[:,1], pos_label=1)
    fnr = 1 - tpr 
    tnr = 1 - fpr
    best_threshold = np.argmax(tpr + tnr)
    print("Best threshold is: ", thresholds[best_threshold])

    tnr_list = list(tnr)

    picked_tnrs = []
    picked_tprs = []
    thresholds_margintest = []
    for tnr_one in tnr_for_margintest:
        picked_tnr = list([0 if x< tnr_one else x for x in tnr_list])
        picked_tnr_threshold = np.argmax(tpr + picked_tnr)        
        thresholds_margintest.append(thresholds[picked_tnr_threshold])
        picked_tnrs.append(np.round(tnr[picked_tnr_threshold], decimals=4))
        picked_tprs.append(np.round(tpr[picked_tnr_threshold], decimals=4))
    # print("TNRS: ", picked_tnrs)
    # print("TPRS: ", picked_tprs)
    # print("Selected Thresholds: ", thresholds_margintest)
    
    target_stack = torch.stack(final_target_list)
    for margin in margin_list:
        for threshold_idx, threshold in enumerate(thresholds_margintest):
            pred_stack = torch.stack(probability_list)
            pred_stack = (pred_stack > threshold).int()
            print("1: ", pred_stack.shape)
            print("2: ", target_stack.shape)
            rise_true, rise_pred_correct, fall_true, fall_pred_correct = binary_detector_evaluator(pred_stack, target_stack, margin)
            print("Margin: {}, Threshold: {}, TPR: {}, TNR: {}".format(str(margin), str(threshold), str(picked_tprs[threshold_idx]), str(picked_tnrs[threshold_idx])))
            # print("rise_t:{}, rise_cor:{}, fall_t:{}, fall_cor:{}".format(str(rise_true), str(rise_pred_correct), str(fall_true), str(fall_pred_correct)))    
            print("rise_accuarcy:{}, fall_accuracy:{}".format(str(np.round((rise_pred_correct/float(rise_true)), decimals=4)), str(np.round((fall_pred_correct/float(fall_true)), decimals=4))))

    return (
        result, 
        np.round(tpr[best_threshold], decimals=4), 
        np.round(fnr[best_threshold], decimals=4), 
        np.round(tnr[best_threshold], decimals=4), 
        np.round(fpr[best_threshold], decimals=4)
    )


def eeg_margin_val_fn(
        hyps, 
        refs, 
        thresholds_margintest, 
        margin_list):
    """Computes metrics for ranking tasks.

    Args:
        qrels: Ground truth. A dictionary of query ids and their corresponding
            relevance judgements. The relevance judgements are a dictionary of
            document ids and their corresponding relevance scores.
        results: Ranked results. A dictionary of query ids and their corresponding
            document scores. The document scores are a dictionary of document ids and
            their corresponding scores.
        k_values: A list of integers specifying the cutoffs for the metrics.

    Returns:
        A dictionary of metrics and their corresponding values.

    Examples:
        >>> qrels = {
        ...     "q1": {"d1": 1, "d2": 0, "d3": 1},
        ...     "q2": {"d1": 1, "d2": 1, "d3": 0}
        ... }
        >>> results = {
        ...     "q1": {"d1": 0.5, "d2": 0.2, "d3": 0.1},
        ...     "q2": {"d1": 0.1, "d2": 0.2, "d3": 0.5}
        ... }
        >>> k_values = [1, 2]
        >>> ranking_metrics_fn(qrels, results, k_values)
        {'NDCG@1': 0.5, 'MAP@1': 0.25, 'Recall@1': 0.25, 'P@1': 0.5, 'NDCG@2': 0.5, 'MAP@2': 0.375, 'Recall@2': 0.5, 'P@2': 0.5}
    """

    margin_3sec_rise_seeds = []
    margin_3sec_fall_seeds = []
    margin_5sec_rise_seeds = []
    margin_5sec_fall_seeds = []

    hyps_list = [list(hyp) for hyp in hyps]
    print("##### margin test evaluation #####")
    target_stack = torch.tensor([item for sublist in refs for item in sublist])
    print("thresholds_margintest: ", thresholds_margintest)
    for margin in margin_list:
        for threshold_idx, threshold in enumerate(thresholds_margintest):
            hyp_output = list([[int(hyp_step > threshold) for hyp_step in hyp_one] for hyp_one in hyps_list])
            pred_stack = torch.tensor(list([item for sublist in hyp_output for item in sublist]))
            pred_stack2 = pred_stack.unsqueeze(1)
            target_stack2 = target_stack.unsqueeze(1)
            rise_true, rise_pred_correct, fall_true, fall_pred_correct = binary_detector_evaluator(pred_stack2, target_stack2, margin)
            # print("Margin: {}, Threshold: {}, TPR: {}, TNR: {}".format(str(margin), str(threshold), str(logger.evaluator.picked_tprs[threshold_idx]), str(logger.evaluator.picked_tnrs[threshold_idx])))
            print("rise_accuarcy:{}, fall_accuracy:{}".format(str(np.round((rise_pred_correct/float(rise_true)), decimals=4)), str(np.round((fall_pred_correct/float(fall_true)), decimals=4))))
            if margin == 3:
                margin_3sec_rise_seeds.append(np.round((rise_pred_correct/float(rise_true)), decimals=4))
                margin_3sec_fall_seeds.append(np.round((fall_pred_correct/float(fall_true)), decimals=4))

            if margin == 5:
                margin_5sec_rise_seeds.append(np.round((rise_pred_correct/float(rise_true)), decimals=4))
                margin_5sec_fall_seeds.append(np.round((fall_pred_correct/float(fall_true)), decimals=4))
                
    return (
        margin_3sec_rise_seeds,
        margin_3sec_fall_seeds,
        margin_5sec_rise_seeds,
        margin_5sec_fall_seeds,
    )


def binary_detector_evaluator(pred_stack, target_stack, margin):
    rise_true, rise_pred_correct, fall_true, fall_pred_correct = 0, 0, 0, 0
    target_rotated = torch.cat([target_stack[0].unsqueeze(0), target_stack[:-1]], dim=0)
    pred_rotated = torch.cat([pred_stack[0].unsqueeze(0), pred_stack[:-1]], dim=0)

    # -1 is at where label goes 0 to 1 (at point of 1)
    # 1 is at where label goes 1 to 0 (at point of 0)
    target_change = torch.subtract(target_rotated, target_stack) 
    pred_change = torch.subtract(pred_rotated, pred_stack) 

    # total_target_fall = (target_change == 1).sum()
    # total_target_rise = (target_change == -1).sum()
    
    for idx, sample in enumerate(target_change.permute(1,0)):
        fall_index_list = (sample == 1).nonzero(as_tuple=True)[0]
        rise_index_list = (sample == -1).nonzero(as_tuple=True)[0]

        for fall_index in fall_index_list:
            start_margin_index = fall_index - margin
            end_margin_index = fall_index + margin
            if start_margin_index < 0:
                start_margin_index = 0
            if end_margin_index > len(sample):
                end_margin_index = len(sample)
            if 1 in pred_change[start_margin_index:end_margin_index+1]:
                fall_pred_correct += 1
            fall_true += 1
        for rise_index in rise_index_list:
            start_margin_index = rise_index - margin
            end_margin_index = rise_index + margin
            if start_margin_index < 0:
                start_margin_index = 0
            if end_margin_index > len(sample):
                end_margin_index = len(sample)
            if -1 in pred_change[start_margin_index:end_margin_index+1]:
                rise_pred_correct += 1
            rise_true += 1
    
    return rise_true, rise_pred_correct, fall_true, fall_pred_correct



if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
