import numpy as np
def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    # Write code here
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    TP = FP = FN = 0
    for label in labels:
        TP += np.sum((y_true == label) & (y_pred == label))
        FP += np.sum((y_true != label) & (y_pred == label))
        FN += np.sum((y_true == label) & (y_pred != label))
    return 2 * TP / (2 * TP + FP + FN)