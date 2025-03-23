import numpy as np

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred, average='binary'):
    if average == 'binary':
        tp = np.sum((y_true == 1) & (y_pred == 1))
        pred_pos = np.sum(y_pred == 1)
        return tp / pred_pos if pred_pos != 0 else 0

    elif average == 'macro':
        classes = np.unique(np.concatenate([y_true, y_pred]))
        precisions = []
        for cls in classes:
            tp = np.sum((y_pred == cls) & (y_true == cls))
            fp = np.sum((y_pred == cls) & (y_true != cls))
            denom = tp + fp
            precisions.append(tp / denom if denom != 0 else 0)
        return np.mean(precisions)


def recall(y_true, y_pred, average='binary'):
    if average == 'binary':
        tp = np.sum((y_true == 1) & (y_pred == 1))
        actual_pos = np.sum(y_true == 1)
        return tp / actual_pos if actual_pos != 0 else 0

    elif average == 'macro':
        classes = np.unique(np.concatenate([y_true, y_pred]))
        recalls = []
        for cls in classes:
            tp = np.sum((y_pred == cls) & (y_true == cls))
            fn = np.sum((y_pred != cls) & (y_true == cls))
            denom = tp + fn
            recalls.append(tp / denom if denom != 0 else 0)
        return np.mean(recalls)


def f1_score(y_true, y_pred, average='binary'):
    if average == 'binary':
        p = precision(y_true, y_pred, average='binary')
        r = recall(y_true, y_pred, average='binary')
        return 2 * p * r / (p + r) if (p + r) != 0 else 0

    elif average == 'macro':
        classes = np.unique(np.concatenate([y_true, y_pred]))
        f1s = []
        for cls in classes:
            p = precision((y_true == cls).astype(int), (y_pred == cls).astype(int))
            r = recall((y_true == cls).astype(int), (y_pred == cls).astype(int))
            f1 = 2 * p * r / (p + r) if (p + r) != 0 else 0
            f1s.append(f1)
        return np.mean(f1s)


def confusion_matrix(y_true, y_pred):
    # Map class labels (including negatives) to 0-based indices
    classes = np.unique(np.concatenate([y_true, y_pred]))
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    index_to_class = {idx: cls for cls, idx in class_to_index.items()}

    matrix = np.zeros((len(classes), len(classes)), dtype=int)

    for t, p in zip(y_true, y_pred):
        i = class_to_index[t]
        j = class_to_index[p]
        matrix[i][j] += 1

    return matrix


def mean_squared_error(y_true, y_pred):
    errors = y_true - y_pred
    return np.mean(errors ** 2)
