import numpy as np
from sklearn.cross_validation import KFold

from settings import *

##################### HELPERS ############################
def load_data(path, count, with_labels=False):
    """
    :param count: feature count to load for one subset
    :param with_labels: indicate whether labels are present
    :rtype: 2/3-element tuple - [labels], first_feature_array, second_feature_array
    """
    labels, a_features, b_features = [], [], []
    with open(path) as f:
        offset = 1 if with_labels else 0
        headers = f.next().rstrip().split(',')
        for line in f:
            values = line.rstrip().split(',')
            if with_labels:
                labels.append(values[0])
            a_features.append([float(x) for x in values[offset:count+offset]])
            b_features.append([float(x) for x in values[count+offset:]])
        if with_labels:
            return np.array(labels), np.array(a_features), np.array(b_features)
        else:
            return np.array(a_features), np.array(b_features)


################# TRAINING #######################
def predict(a_features, b_features):
    """
    Magical functions that makes all predictions.

    return: A user is more influential than B
    rtype: bool
    """
    pass


#################### RESULTS ######################
def produce_results(fname):
    with open(fname, 'w+') as f:
        af, bf = load_data(TEST_PATH, 11)
        for a_features, b_features in zip(list(af), list(bf)):
            a_better = predict(a_features, b_features)
            f.write(bool(a_better)) # label '1' means A is more influential than B. 0 means B is more influential than A


#################### MAIN ##########################
if __name__ == "__main__":
    # Load data
    labels, A, B = load_data(TRAIN_PATH, 11, with_labels=True)

    # split original set into training and testing
    kf = KFold(len(A), 2)
    for train_indices, test_indices in kf:
        train_labels, train_A, train_B = labels[train_indices], A[train_indices], B[train_indices]
        # learn model using training data

        test_labels, test_A, test_B = labels[test_indices], A[test_indices], B[test_indices]
        # test model using test data

    # produce results
    produce_results('submission.csv')