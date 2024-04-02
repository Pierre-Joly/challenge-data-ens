import numpy as np

def aligner_formes(a, b):
    if a.shape != b.shape:
        if a.shape[0] == b.shape[1] and a.shape[1] == b.shape[0]:
            b = np.transpose(b)
        else:
            raise ValueError("Les formes des vecteurs ne sont pas compatibles.")
    return a, b

def weighted_accuracy(y_pred, y_test):
    y_test = np.array(y_test)[:,0]
    y_pred, y_test = aligner_formes(y_pred, y_test)
    correct_predictions_proportion = (np.sign(y_pred) * y_test > 0).astype(int)
    weights = np.abs(y_test-y_pred)
    return np.sum(correct_predictions_proportion * weights) / np.sum(weights)

