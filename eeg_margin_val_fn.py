
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
