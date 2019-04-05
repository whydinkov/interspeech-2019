def calc_mae(y_true, y_pred):
    mae = 0

    if len(y_true) != len(y_pred):
        raise Exception('Cannot compare different length lists.')

    for true_label, pred_label in zip(y_true, y_pred):
        mae += _calc_label_error(true_label, pred_label)

    return mae / len(y_true)


def _calc_label_error(true_label, pred_label):
    # no error
    if true_label == pred_label:
        return 0

    # pred will be left or right
    # so error is 1
    if true_label == 'leastbiased':
        return 1

    # if true is left or right
    # mistaken by center is 1
    if pred_label == 'leastbiased':
        return 1

    # otherwise 2
    return 2
